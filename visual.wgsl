struct Global
{
  width: u32,
  height: u32,
  samplesPerPixel: u32,
  maxRecursion: u32,
  randSeed: vec3f,
  weight: f32,
  time: f32,
  pad1: f32,
  pad2: f32,
  pad3: f32,
  eye: vec3f,
  vertFov: f32,
  right: vec3f,
  focDist: f32,
  up: vec3f,
  focAngle: f32,
  pixelDeltaX: vec3f,
  pad4: f32,
  pixelDeltaY: vec3f,
  pad5: f32,
  pixelTopLeft: vec3f,
  pad6: f32
}

struct Ray
{
  ori: vec3f,
  dir: vec3f
}

struct Hit
{
  dist: f32,
  pos: vec3f,
  nrm: vec3f,
  inside: bool,
  matOfs: u32
}

const EPSILON = 0.001;
const PI = 3.141592;
const MAX_DIST = 3.402823466e+38;

const OBJ_TYPE_SPHERE = 0;
const OBJ_TYPE_PLANE = 1;
const OBJ_TYPE_BOX = 2;
const OBJ_TYPE_CYLINDER = 3;
const OBJ_TYPE_MESH = 4;

const MAT_TYPE_LAMBERT = 0;
const MAT_TYPE_METAL = 1;
const MAT_TYPE_GLASS = 2;

@group(0) @binding(0) var<uniform> globals: Global;
@group(0) @binding(1) var<storage, read> objects: array<f32>;
@group(0) @binding(2) var<storage, read> materials: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> image: array<vec4f>;

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
  let center = vec3f(objects[objOfs + 1], objects[objOfs + 2], objects[objOfs + 3]);
  let radius = objects[objOfs + 4];

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

  (*h).dist = t;
  (*h).pos = rayPos(r, t);
  (*h).nrm = ((*h).pos - center) / radius;
  (*h).inside = dot(r.dir, (*h).nrm) > 0;
  (*h).nrm *= select(1.0, -1.0, (*h).inside);
  (*h).matOfs = u32(objects[objOfs + 5]);

  return true;
}

fn evalMaterialLambert(in: Ray, h: Hit, att: ptr<function, vec3f>, out: ptr<function, Ray>) -> bool
{
  let dir = h.nrm + rand3UnitSphere(); 
  *out = Ray(h.pos, select(normalize(dir), h.nrm, all(abs(dir) < vec3f(EPSILON))));
  *att = vec3f(materials[h.matOfs + 1], materials[h.matOfs + 2], materials[h.matOfs + 3]);
  return true;
}

fn evalMaterialMetal(in: Ray, h: Hit, att: ptr<function, vec3f>, out: ptr<function, Ray>) -> bool
{
  let fuzzRadius = materials[h.matOfs + 4];
  let dir = reflect(normalize(in.dir), h.nrm);
  *out = Ray(h.pos, dir + fuzzRadius * rand3UnitSphere());
  *att = vec3f(materials[h.matOfs + 1], materials[h.matOfs + 2], materials[h.matOfs + 3]);
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
  let refractionIndex = materials[h.matOfs + 4];
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
  *att = vec3f(materials[h.matOfs + 1], materials[h.matOfs + 2], materials[h.matOfs + 3]);
  return true;
}

fn evalMaterial(in: Ray, h: Hit, att: ptr<function, vec3f>, out: ptr<function, Ray>) -> bool
{
  switch(u32(materials[h.matOfs]))
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
      // TODO How to indicate an error?
      return false;
    }
  }
}

fn intersectScene(ray: Ray, tmin: f32, tmax: f32, hit: ptr<function, Hit>) -> bool
{
  var objOfs = 0u;
  (*hit).dist = tmax;

  loop {  
    switch(u32(objects[objOfs])) {
      case OBJ_TYPE_SPHERE: {
        intersectSphere(ray, tmin, (*hit).dist, objOfs, hit);
        objOfs += 6;
      }
      case OBJ_TYPE_PLANE: {
        objOfs += 0;
      }
      default: {
        // Jump beyond end of data on error
        objOfs += 99999;
      }
    }
    if(objOfs >= arrayLength(&objects)) {
      break;
    }
  }

  return (*hit).dist < tmax;
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
    if(depth >= globals.maxRecursion) {
      break;
    }
  }

  return col;
}

fn createPrimaryRay(pixelPos: vec2f) -> Ray
{
  var pixelSample = globals.pixelTopLeft + globals.pixelDeltaX * pixelPos.x + globals.pixelDeltaY * pixelPos.y;
  pixelSample += (rand() - 0.5) * globals.pixelDeltaX + (rand() - 0.5) * globals.pixelDeltaY;

  var eyeSample = globals.eye;
  if(globals.focAngle > 0) {
    let focRadius = globals.focDist * tan(0.5 * radians(globals.focAngle));
    let diskSample = rand2Disk();
    eyeSample += focRadius * (diskSample.x * globals.right + diskSample.y * globals.up);
  }

  return Ray(eyeSample, normalize(pixelSample - eyeSample));
}

@compute @workgroup_size(8,8)
fn computeMain(@builtin(global_invocation_id) globalId: vec3u)
{
  if(all(globalId.xy >= vec2u(globals.width, globals.height))) {
    return;
  }
  
  initRand(globalId, vec3u(globals.randSeed * 0xffffffff));

  var col = vec3f(0);
  for(var i=0u; i<globals.samplesPerPixel; i++) {
    col += render(createPrimaryRay(vec2f(globalId.xy)), MAX_DIST);
  }

  let index = globals.width * globalId.y + globalId.x;
  let outCol = mix(buffer[index].xyz, col / f32(globals.samplesPerPixel), globals.weight);

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
  return image[globals.width * u32(pos.y) + u32(pos.x)];
}
