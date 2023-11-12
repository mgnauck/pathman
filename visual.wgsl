struct Uniforms
{
  width: f32,
  height: f32,
  samplesPerPixel: f32,
  maxRecursion: f32,
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
  matType: u32,
  matId: u32
}

struct Sphere
{
  center: vec3f,
  radius: f32,
  matType: u32,
  matId: u32
}

struct Box
{
  // TODO
  matType: u32,
  matId: u32
}

struct LambertMaterial
{
  albedo: vec3f
}

struct MetalMaterial
{
  albedo: vec3f,
  fuzzRadius: f32
}

struct GlassMaterial
{
  albedo: vec3f,
  refractionIndex: f32
}

const epsilon = 0.001;
const pi = 3.141592;
const maxDist = 3.402823466e+38;

const lambertMaterialCount = 2;
var<private> lambertMaterials = array<LambertMaterial, lambertMaterialCount>(
  LambertMaterial(vec3f(0.5)),
  LambertMaterial(vec3f(0.6, 0.3, 0.3))
);

const metalMaterialCount = 1;
var<private> metalMaterials = array<MetalMaterial, metalMaterialCount>(
  MetalMaterial(vec3f(0.3, 0.3, 0.6), 0.0)
);

const glassMaterialCount = 1;
var<private> glassMaterials = array<GlassMaterial, glassMaterialCount>(
  GlassMaterial(vec3f(1.0), 1.5)
);

const sphereCount = 5;
var<private> spheres = array<Sphere, sphereCount>(
  Sphere(vec3f(0, -100.5, 0), 100, 0, 0),
  Sphere(vec3f(-1, 0, 0), 0.5, 0, 1),
  Sphere(vec3f(0, 0.0, 0), 0.5, 2, 0),
  Sphere(vec3f(0, 0.0, 0), -0.35, 2, 0),
  Sphere(vec3f(1, 0, 0), 0.5, 1, 0)
);

@group(0) @binding(0) var<uniform> global: Uniforms;
@group(0) @binding(1) var<storage, read_write> buffer: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> image: array<vec4f>;

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
fn rand3OnUnit() -> vec3f
{
  let u = 2 * rand() - 1;
  let theta = 2 * pi * rand();
  let r = sqrt(1 - u * u);
  return vec3f(r * cos(theta), r * sin(theta), u);
}

fn rand3Hemi(nrm: vec3f) -> vec3f
{
  let v = rand3OnUnit();
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
    theta = (pi / 4) * (uOffset.y / uOffset.x);
  } else {
    r = uOffset.y;
    theta = (pi / 2) - (pi / 4) * (uOffset.x / uOffset.y);
  }
  
  return r * vec2f(cos(theta), sin(theta));
}

fn rayPos(r: Ray, t: f32) -> vec3f
{
  return r.ori + t * r.dir;
}

fn intersect(s: Sphere, r: Ray, tmin: f32, tmax: f32, h: ptr<function, Hit>) -> bool
{
  let oc = r.ori - s.center;
  let a = dot(r.dir, r.dir);
  let b = dot(oc, r.dir); // half
  let c = dot(oc, oc) - s.radius * s.radius;

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
  (*h).pos = r.ori + t * r.dir;
  (*h).nrm = ((*h).pos - s.center) / s.radius;
  (*h).inside = dot(r.dir, (*h).nrm) > 0;
  (*h).nrm *= select(1.0, -1.0, (*h).inside);
  (*h).matType = s.matType;
  (*h).matId = s.matId;

  return true;
}

fn evalMaterialLambert(in: Ray, h: Hit, att: ptr<function, vec3f>, out: ptr<function, Ray>) -> bool
{
  let dir = h.nrm + rand3OnUnit(); 
  *out = Ray(h.pos, select(dir, h.nrm, all(abs(dir) < vec3f(epsilon))));
  *att = lambertMaterials[h.matId].albedo;
  return true;
}

fn evalMaterialMetal(in: Ray, h: Hit, att: ptr<function, vec3f>, out: ptr<function, Ray>) -> bool
{
  let mat = &metalMaterials[h.matId];
  let dir = reflect(normalize(in.dir), h.nrm);
  *out = Ray(h.pos, dir + (*mat).fuzzRadius * rand3OnUnit());
  *att = (*mat).albedo;
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
  let mat = &glassMaterials[h.matId];
  let refracIndexRatio = select(1 / (*mat).refractionIndex, (*mat).refractionIndex, h.inside);
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
  *att = (*mat).albedo;
  return true;
}

fn evalMaterial(in: Ray, h: Hit, att: ptr<function, vec3f>, out: ptr<function, Ray>) -> bool
{
  switch(h.matType)
  {
    case 1: {
      return evalMaterialMetal(in, h, att, out);
    }
    case 2: {
      return evalMaterialGlass(in, h, att, out);
    }
    default: {
      return evalMaterialLambert(in, h, att, out);
    }
  }
}

fn intersectPrimitives(r: Ray, tmin: f32, tmax: f32, h: ptr<function, Hit>) -> bool
{
  var currMinDist = tmax;
  for(var i=0u; i<sphereCount; i++) {
    var tempHit: Hit;
    if(intersect(spheres[i], r, tmin, currMinDist, &tempHit)) {
      currMinDist = tempHit.t;
      (*h) = tempHit;
    }
  }
  return currMinDist < tmax;
}

fn render(ray: Ray, maxDist: f32) -> vec3f
{
  var h: Hit;
  var r = ray;
  var d = u32(global.maxRecursion);
  var c = vec3f(1);

  loop {
    if(intersectPrimitives(r, 0.001, maxDist, &h)) {
      var att: vec3f;
      var s: Ray;
      if(evalMaterial(r, h, &att, &s)) {
        c *= att;
        r = s;
      }
    } else {
      let t = (normalize(r.dir).y + 1.0) * 0.5;
      c *= (1.0 - t) * vec3f(1.0) + t * vec3f(0.5, 0.7, 1.0);
      break;
    }

    d -= 1;
    if(d <= 0) {
      break;
    }
  }

  return c;
}

fn makePrimaryRay(pixelPos: vec2f) -> Ray
{
  let viewportHeight = 2 * tan(radians(0.5 * global.vertFov)) * global.focDist;
  let viewportWidth = viewportHeight * global.width / global.height;

  let viewportRight = global.right * viewportWidth; 
  let viewportDown = -global.up * viewportHeight;

  let pixelDeltaX = viewportRight / global.width;
  let pixelDeltaY = viewportDown / global.height;

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

  return Ray(originSample, pixelSample - originSample);
}

@compute @workgroup_size(8,8)
fn computeMain(@builtin(global_invocation_id) globalId: vec3u)
{
  if(all(globalId.xy >= vec2u(u32(global.width), u32(global.height)))) {
    return;
  }
  
  initRand(globalId, vec3u(global.randSeed * 0xffffffff));

  var col = vec3f(0);
  for(var i=0u; i<u32(global.samplesPerPixel); i++) {
    col += render(makePrimaryRay(vec2f(globalId.xy)), maxDist);
  }

  // Accumulation buffer is linear
  let index = u32(global.width) * globalId.y + globalId.x;
  let outCol = mix(buffer[index].xyz, col / global.samplesPerPixel, global.weight);
  buffer[index] = vec4f(outCol, 1);

  // Current output is gamma corrected
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
