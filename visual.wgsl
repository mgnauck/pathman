struct Global
{
  width: u32,
  height: u32,
  samplesPerPixel: u32,
  maxRecursion: u32,
  randSeed: vec3f,
  weight: f32,
  eye: vec3f,
  vertFov: f32,
  right: vec3f,
  focDist: f32,
  up: vec3f,
  focAngle: f32,
  fwd: vec3f,
  time: f32
}

struct Scene
{
  objCnt: u32,
  matArrOfs: u32,
  arr: array<f32>
}

struct Ray
{
  ori: vec3f,
  dir: vec3f
}

struct Hit
{
  t: f32,
  pos: vec3f,
  nrm: vec3f,
  inside: bool,
  matOfs: u32
}

const EPSILON = 0.001;
const PI = 3.141592;
const MAX_DIST = 3.402823466e+38;

const OBJ_TYPE_PLANE = 1;
const OBJ_TYPE_SPHERE = 2;
const OBJ_TYPE_BOX = 3;
const OBJ_TYPE_CYLINDER = 4;
const OBJ_TYPE_MESH = 5;

const MAT_TYPE_LAMBERT = 1;
const MAT_TYPE_METAL = 2;
const MAT_TYPE_GLASS = 3;

@group(0) @binding(0) var<uniform> global: Global;
@group(0) @binding(1) var<storage, read> scene: Scene;
@group(0) @binding(2) var<storage, read_write> buffer: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> image: array<vec4f>;

var<private> seed: vec3u;

// PRNG taken from WebGPU samples
fn initRand(invocationId: vec3u, initialSeed: vec3u)
{
  const A = vec3(1741651 * 1009,
                 140893  * 1609 * 13,
                 6521    * 983  * 7 * 2);
  seed = (invocationId * A) ^ initialSeed;
}

fn rand() -> f32
{
  const C = vec3(60493  * 9377,
                 11279  * 2539 * 23,
                 7919   * 631  * 5 * 3);
  seed = (seed * C) ^ (seed.yzx >> vec3(4u));
  return f32(seed.x ^ seed.y) / f32(0xffffffff);
}

fn randRange(valueMin: f32, valueMax: f32) -> f32
{
  return mix(valueMin, valueMax, rand());
}

fn rand3() -> vec3f
{
  return vec3f(rand(), rand(), rand());
}

fn rand3Range(valueMin: f32, valueMax: f32) -> vec3f
{
  return vec3f(randRange(valueMin, valueMax), randRange(valueMin, valueMax), randRange(valueMin, valueMax));
}

// https://mathworld.wolfram.com/SpherePointPicking.html
fn rand3UnitSphere() -> vec3f
{
  let u = 2 * rand() - 1;
  let theta = 2 * PI * rand();
  let r = sqrt(1 - u * u);
  return vec3f(r * cos(theta), r * sin(theta), u);
}

fn rand3Hemi(nrm: vec3f) -> vec3f
{
  let v = rand3UnitSphere();
  return select(-v, v, dot(v, nrm) > 0);
}

fn rand2Disk() -> vec2f
{
  let u = vec2f(rand(), rand());
  let uOffset = 2 * u - vec2f(1, 1);

  if(uOffset.x == 0 && uOffset.y == 0) {
    return vec2f(0);
  }
  
  var theta = 0.0;
  var r = 0.0;
  if(abs(uOffset.x) > abs(uOffset.y)) {
    r = uOffset.x;
    theta = (PI / 4) * (uOffset.y / uOffset.x);
  } else {
    r = uOffset.y;
    theta = (PI / 2) - (PI / 4) * (uOffset.x / uOffset.y);
  }
  
  return r * vec2f(cos(theta), sin(theta));
}

fn rayPos(r: Ray, dist: f32) -> vec3f
{
  return r.ori + dist * r.dir;
}

fn intersectSphere(r: Ray, tmin: f32, tmax: f32, objOfs: u32, h: ptr<function, Hit>) -> bool
{
  let center = vec3f(scene.arr[objOfs + 1], scene.arr[objOfs + 2], scene.arr[objOfs + 3]);
  let radius = scene.arr[objOfs + 4];

  let oc = r.ori - center;
  let a = dot(r.dir, r.dir);
  let b = dot(oc, r.dir); // half
  let c = dot(oc, oc) - radius * radius;

  let d = b * b - a * c;
  if(d < 0) {
    return false;
  }

  let sqrtd = sqrt(d);
  var t = (-b - sqrtd) / a;
  if(t <= tmin || tmax <= t) {
    t = (-b + sqrtd) / a;
    if(t <= tmin || tmax <= t) {
      return false;
    }
  }

  (*h).t = t;
  (*h).pos = rayPos(r, t);
  (*h).nrm = ((*h).pos - center) / radius;
  (*h).inside = dot(r.dir, (*h).nrm) > 0;
  (*h).nrm *= select(1.0, -1.0, (*h).inside);
  (*h).matOfs = scene.matArrOfs + u32(scene.arr[objOfs + 5]);

  return true;
}

fn evalMaterialLambert(in: Ray, h: Hit, att: ptr<function, vec3f>, out: ptr<function, Ray>) -> bool
{
  let dir = h.nrm + rand3UnitSphere(); 
  *out = Ray(h.pos, select(normalize(dir), h.nrm, all(abs(dir) < vec3f(EPSILON))));
  *att = vec3f(scene.arr[h.matOfs + 1], scene.arr[h.matOfs + 2], scene.arr[h.matOfs + 3]);
  return true;
}

fn evalMaterialMetal(in: Ray, h: Hit, att: ptr<function, vec3f>, out: ptr<function, Ray>) -> bool
{
  let fuzzRadius = scene.arr[h.matOfs + 4];
  let dir = reflect(normalize(in.dir), h.nrm);
  *out = Ray(h.pos, dir + fuzzRadius * rand3UnitSphere());
  *att = vec3f(scene.arr[h.matOfs + 1], scene.arr[h.matOfs + 2], scene.arr[h.matOfs + 3]);
  return dot((*out).dir, h.nrm) > 0;
}

fn schlickReflectance(cosTheta: f32, refractionIndexRatio: f32) -> f32
{
  var r0 = (1 - refractionIndexRatio) / (1 + refractionIndexRatio);
  r0 = r0 * r0;
  return r0 + (1 - r0) * pow(1 - cosTheta, 5);
}

fn evalMaterialGlass(in: Ray, h: Hit, att: ptr<function, vec3f>, out: ptr<function, Ray>) -> bool
{
  let refractionIndex = scene.arr[h.matOfs + 4];
  let refracIndexRatio = select(1 / refractionIndex, refractionIndex, h.inside);
  let inDir = normalize(in.dir);
  
  let cosTheta = min(dot(-inDir, h.nrm), 1);
  /*let sinTheta = sqrt(1 - cosTheta * cosTheta);

  var dir: vec3f;
  if(refracIndexRatio * sinTheta > 1 || schlickReflectance(cosTheta, refracIndexRatio) > rand()) {
    dir = reflect(inDir, h.nrm);
  } else {
    dir = refract(inDir, h.nrm, refracIndexRatio);
  }*/

  var dir = refract(inDir, h.nrm, refracIndexRatio);
  if(all(dir == vec3f(0)) || schlickReflectance(cosTheta, refracIndexRatio) > rand()) {
    dir = reflect(inDir, h.nrm);
  }

  *out = Ray(h.pos, dir);
  *att = vec3f(scene.arr[h.matOfs + 1], scene.arr[h.matOfs + 2], scene.arr[h.matOfs + 3]);
  return true;
}

fn evalMaterial(in: Ray, h: Hit, att: ptr<function, vec3f>, out: ptr<function, Ray>) -> bool
{
  switch(u32(scene.arr[h.matOfs]))
  {
    case MAT_TYPE_LAMBERT: { 
      return evalMaterialLambert(in, h, att, out);
    }
    case MAT_TYPE_METAL: {
      return evalMaterialMetal(in, h, att, out);
    }
    case MAT_TYPE_GLASS: { 
      return evalMaterialGlass(in, h, att, out);
    }
    default: {
      return false;
    }
  }
}

fn intersect(ray: Ray, tmin: f32, tmax: f32, objOfs: ptr<function, u32>, hit: ptr<function, Hit>) -> bool
{
  //*objOfs += 6;
  //return intersectSphere(ray, tmin, tmax, *objOfs - 6, hit);
 
  switch(u32(scene.arr[*objOfs]))
  {
    case OBJ_TYPE_SPHERE: { 
      *objOfs += 6;
      return intersectSphere(ray, tmin, tmax, *objOfs - 6, hit);
    }
    case OBJ_TYPE_PLANE: {
      return false;
    }
    default: {
      return false;
    }
  }
}

fn intersectScene(ray: Ray, tmin: f32, tmax: f32, hit: ptr<function, Hit>) -> bool
{
  var currMinDist = tmax;
  var currObjOfs = 0u;
  for(var i=0u; i<scene.objCnt; i++) {
    var tempHit: Hit;
    if(intersect(ray, tmin, currMinDist, &currObjOfs, &tempHit)) {
      currMinDist = tempHit.t;
      (*hit) = tempHit;
    }
  }
  return currMinDist < tmax;
}

fn sampleBackground(ray: Ray) -> vec3f
{
  let t = (normalize(ray.dir).y + 1.0) * 0.5;
  return (1.0 - t) * vec3f(1.0) + t * vec3f(0.5, 0.7, 1.0);
}

fn render(ray: Ray, MAX_DIST: f32) -> vec3f
{
  var h: Hit;
  var r = ray;
  var depth = 0u;
  var col = vec3f(1);

  loop {
    if(intersectScene(r, EPSILON, MAX_DIST, &h)) {
      var att: vec3f;
      var s: Ray;
      if(evalMaterial(r, h, &att, &s)) {
        col *= att;
        r = s;
      }
    } else {
      col *= sampleBackground(r);
      break;
    }

    depth += 1;
    if(depth >= global.maxRecursion) {
      break;
    }
  }

  return col;
}

fn makePrimaryRay(pixelPos: vec2f) -> Ray
{
  let width = f32(global.width);
  let height = f32(global.height);

  let viewportHeight = 2 * tan(radians(0.5 * global.vertFov)) * global.focDist;
  let viewportWidth = viewportHeight * width / height;

  let viewportRight = global.right * viewportWidth; 
  let viewportDown = -global.up * viewportHeight;

  let pixelDeltaX = viewportRight / width;
  let pixelDeltaY = viewportDown / height;

  let viewportTopLeft = global.eye - global.focDist * global.fwd - 0.5 * (viewportRight + viewportDown);
  let pixelTopLeft = viewportTopLeft + 0.5 * (pixelDeltaX + pixelDeltaY);

  var pixelSample = pixelTopLeft + pixelDeltaX * pixelPos.x + pixelDeltaY * pixelPos.y;
  pixelSample += (rand() - 0.5) * pixelDeltaX + (rand() - 0.5) * pixelDeltaY;

  var originSample = global.eye;
  if(global.focAngle > 0) {
    let focRadius = global.focDist * tan(0.5 * radians(global.focAngle));
    let diskSample = rand2Disk();
    originSample += focRadius * (diskSample.x * global.right + diskSample.y * global.up);
  }

  return Ray(originSample, pixelSample - originSample); // normalize dir
}

@compute @workgroup_size(8,8)
fn computeMain(@builtin(global_invocation_id) globalId: vec3u)
{
  if(all(globalId.xy >= vec2u(global.width, global.height))) {
    return;
  }
  
  initRand(globalId, vec3u(global.randSeed * 0xffffffff));

  let spp = global.samplesPerPixel;
  var col = vec3f(0);
  for(var i=0u; i<spp; i++) {
    col += render(makePrimaryRay(vec2f(globalId.xy)), MAX_DIST);
  }

  let index = global.width * globalId.y + globalId.x;
  let outCol = mix(buffer[index].xyz, col / f32(global.samplesPerPixel), global.weight);

  buffer[index] = vec4f(outCol, 1);
  image[index] = vec4f(pow(outCol, vec3f(0.4545)), 1);
}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f
{
  let pos = array<vec2f, 4>(vec2f(-1, 1), vec2f(-1), vec2f(1), vec2f(1, -1));
  return vec4f(pos[vertexIndex], 0, 1);
}

@fragment
fn fragmentMain(@builtin(position) pos: vec4f) -> @location(0) vec4f
{
  return image[u32(global.width) * u32(pos.y) + u32(pos.x)];
}
