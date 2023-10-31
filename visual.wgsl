struct Uniforms
{
  width: f32,
  height: f32,
  time: f32,
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

struct DielectricMaterial
{
  albedo: vec3f,
  refractionIndex: f32
}

const epsilon = 0.001;
const pi = 3.141592;
const maxDist = 3.402823466e+38;
const samplesPerPixel = 100u;
const maxRecursion = 50u;

const lambertMaterialCount = 2;
var<private> lambertMaterials = array<LambertMaterial, lambertMaterialCount>(
  LambertMaterial(vec3f(0.8, 0.8, 0.0)),
  LambertMaterial(vec3f(0.1, 0.2, 0.5))
);

const metalMaterialCount = 1;
var<private> metalMaterials = array<MetalMaterial, metalMaterialCount>(
  MetalMaterial(vec3f(0.8, 0.6, 0.2), 0.0)
);

const dielectricMaterialCount = 1;
var<private> dielectricMaterials = array<DielectricMaterial, dielectricMaterialCount>(
  DielectricMaterial(vec3f(1.0), 1.5)
);

const sphereCount = 5;
var<private> spheres = array<Sphere, sphereCount>(
  Sphere(vec3f(0, -100.5, -1), 100, 0, 0),
  Sphere(vec3f(-1, 0, -1), 0.5, 0, 1),
  Sphere(vec3f(0, 0, -1), 0.5, 2, 0),
  Sphere(vec3f(0, 0, -1), -0.4, 2, 0),
  Sphere(vec3f(1, 0, -1), 0.5, 1, 0)
);

@group(0) @binding(0) var<uniform> global: Uniforms;
@group(0) @binding(1) var<storage, read_write> buffer: array<vec4f>;

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

fn rand3Unit() -> vec3f
{
  let theta = 2.0 * pi * rand();
  let phi = acos(2.0 * rand() - 1.0);
  let r = pow(rand(), 1.0 / 3.0);
  let sin_phi = sin(phi);
  return r * vec3f(sin_phi * sin(theta), sin_phi * cos(theta), cos(phi));
}

fn rand3Hemi(nrm: vec3f) -> vec3f
{
  let v = rand3Unit();
  return select(-v, v, dot(v, nrm) > 0);
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
  let dir = h.nrm + rand3Unit(); 
  *out = Ray(h.pos, select(dir, h.nrm, all(abs(dir) < vec3f(epsilon))));
  *att = lambertMaterials[h.matId].albedo;
  return true;
}

fn evalMaterialMetal(in: Ray, h: Hit, att: ptr<function, vec3f>, out: ptr<function, Ray>) -> bool
{
  let mat = &metalMaterials[h.matId];
  let dir = reflect(normalize(in.dir), h.nrm);
  *out = Ray(h.pos, dir + (*mat).fuzzRadius * rand3Unit());
  *att = (*mat).albedo;
  return dot((*out).dir, h.nrm) > 0;
}

fn schlickReflectance(cosTheta: f32, refractionIndexRatio: f32) -> f32
{
  var r0 = (1 - refractionIndexRatio) / (1 + refractionIndexRatio);
  r0 = r0 * r0;
  return r0 + (1 - r0) * pow(1 - cosTheta, 5);
}

fn evalMaterialDielectric(in: Ray, h: Hit, att: ptr<function, vec3f>, out: ptr<function, Ray>) -> bool
{
  let mat = &dielectricMaterials[h.matId];
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
      return evalMaterialDielectric(in, h, att, out);
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

fn render(ray: Ray, maxDist: f32, maxRecursion: u32) -> vec3f
{
  var h: Hit;
  var r = ray;
  var d = maxRecursion;
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

fn makePrimaryRay(vertFov: f32, eye: vec3f, tgt: vec3f, vup: vec3f, pixelPos: vec2f) -> Ray
{
  // Right handed coordinate system, with camera looking down negative Z
  var fwd = eye - tgt;

  let focalLen = length(fwd);
  let height = tan(radians(vertFov) / 2);

  let viewportHeight = 2 * height * focalLen;
  let viewportWidth = viewportHeight * global.width / global.height;

  fwd /= focalLen;
  let ri = normalize(cross(vup, fwd));
  let up = cross(fwd, ri);
 
  let viewportRight = ri * viewportWidth;
  let viewportDown = -up * viewportHeight;

  let pixelDeltaX = viewportRight / global.width;
  let pixelDeltaY = viewportDown / global.height;

  let viewportTopLeft = eye - fwd * focalLen - 0.5 * (viewportRight + viewportDown);
  let pixelTopLeft = viewportTopLeft + 0.5 * (pixelDeltaX + pixelDeltaY);

  var pixelTarget = pixelTopLeft + pixelDeltaX * pixelPos.x + pixelDeltaY * pixelPos.y;
  pixelTarget += (rand() - 0.5) * pixelDeltaX + (rand() - 0.5) * pixelDeltaY;

  return Ray(eye, pixelTarget - eye);
}

@compute @workgroup_size(8,8)
fn computeMain(@builtin(global_invocation_id) globalId: vec3u)
{
  if(all(globalId.xy >= vec2u(u32(global.width), u32(global.height)))) {
    return;
  }
  
  initRand(globalId, vec3u(37, 98234, 1236734));

  let index = u32(global.width) * globalId.y + globalId.x;

  /*spheres[1].radius = 0.3 + 0.25 * cos(global.time);
  spheres[2].center.y = sin(global.time);
  spheres[3].center.y = sin(global.time);*/

  var col = vec3f(0);
  for(var i=0u; i<samplesPerPixel; i++) {
    //let ray = makePrimaryRay(50.0, vec3f(-2, 2, 1), vec3f(0, 0, -1), vec3f(0, 1, 0), vec2f(globalId.xy));
    let ray = makePrimaryRay(50.0, vec3f(4 * sin(global.time), 0, 4 * cos(global.time)), vec3f(0, 0, 0), vec3f(0, 1, 0), vec2f(globalId.xy));
    col += render(ray, maxDist, maxRecursion);
  }

  buffer[u32(global.width) * globalId.y + globalId.x] = vec4f(pow(col / f32(samplesPerPixel), vec3f(0.4545)), 1.0);
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
  return buffer[u32(global.width) * u32(pos.y) + u32(pos.x)];
}
