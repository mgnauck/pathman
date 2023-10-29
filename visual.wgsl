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

const epsilon = 0.001;
const pi = 3.141592;
const maxDist = 3.402823466e+38;
const samplesPerPixel = 50u;
const maxRecursion = 5u;

const sphereCount = 4;
var<private> spheres = array<Sphere, sphereCount>(
  Sphere(vec3f(-0.3, 0, -1.2), 0.6, 0, 1),
  Sphere(vec3f(0.3, 0, -1), 0.3, 0, 2),
  Sphere(vec3f(0, 0, -0.7), 0.3, 0, 3),
  Sphere(vec3f(0, -100.5, -1), 100, 0, 0)
);

const lambertMaterialCount = 4;
var<private> lambertMaterials = array<LambertMaterial, lambertMaterialCount>(
  LambertMaterial(vec3f(0.5)),
  LambertMaterial(vec3f(0.3, 0.3, 0.6)),
  LambertMaterial(vec3f(0.3, 0.6, 0.3)),
  LambertMaterial(vec3f(0.6, 0.3, 0.3)),
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
  let b = dot(oc, r.dir);
  let c = dot(oc, oc) - s.radius * s.radius;

  // Origin outside of sphere (c > 0) and dir pointing away from sphere (b >0)
  if(c > 0 && b > 0) {
    return false;
  }

  let d = b * b - c;

  // Negative discriminant = ray missing sphere
  if(d < 0) {
    return false;
  }

  // Solve considering min/max interval and inside sphere situation
  let sd = sqrt(d);
  var t = -b - sd;
  var inside = 1.0;
  if(t <= tmin || tmax <= t) {
    t = -b + sd;
    if(t <= tmin || tmax <= t) {
      return false;
    }
    inside = -1.0;
  }

  (*h).t = t;
  (*h).pos = r.ori + t * r.dir;
  (*h).nrm = inside * ((*h).pos - s.center) / s.radius;
  (*h).matType = s.matType;
  (*h).matId = s.matId;

  return true;
}

fn evalMaterialLambert(r: Ray, h: Hit, att: ptr<function, vec3f>, s: ptr<function, Ray>)
{
  var dir = h.nrm + rand3Hemi(h.nrm); 
  *s = Ray(h.pos, select(normalize(dir), h.nrm, all(abs(dir) < vec3f(epsilon))));
  *att = lambertMaterials[h.matId].albedo;
}

fn evalMaterial(r: Ray, h: Hit, att: ptr<function, vec3f>, s: ptr<function, Ray>)
{
  switch(h.matType)
  {
    // TODO
    default: {
      evalMaterialLambert(r, h, att, s);
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
      evalMaterial(r, h, &att, &s);
      c *= att;
      r = s;
    } else {
      let t = (normalize(r.dir).y + 1.0) * 0.5;
      c *= (1.0 - t) * vec3f(1.0) + t * vec3f(0.3, 0.7, 1.0);
      break;
    }

    d -= 1;
    if(d <= 0) {
      break;
    }
  }

  return c;
}

fn makePrimaryRay(width: f32, height: f32, focalLen: f32, eye: vec3f, pixelPos: vec2f) -> Ray
{
  let viewportRight = vec3f(2 * width / height, 0, 0);
  let viewportDown = vec3f(0, -2, 0);

  let pixelDeltaX = viewportRight / width;
  let pixelDeltaY = viewportDown / height;

  let viewportTopLeft = eye - vec3f(0, 0, focalLen) - 0.5 * (viewportRight + viewportDown);
  let pixelTopLeft = viewportTopLeft + 0.5 * (pixelDeltaX + pixelDeltaY);

  var pixelTarget = pixelTopLeft + pixelDeltaX * pixelPos.x + pixelDeltaY * pixelPos.y;
  pixelTarget += (rand() - 0.5) * pixelDeltaX + (rand() - 0.5) * pixelDeltaY;

  return Ray(eye, normalize(pixelTarget - eye));
}

@compute @workgroup_size(8,8)
fn computeMain(@builtin(global_invocation_id) globalId: vec3u)
{
  if(all(globalId.xy >= vec2u(u32(global.width), u32(global.height)))) {
    return;
  }
  
  initRand(globalId, vec3u(37, 98234, 1236734));

  let index = u32(global.width) * globalId.y + globalId.x;

  spheres[0].center.y = sin(global.time * 0.7) * 0.5;
  spheres[1].center.y = cos(global.time * 0.5) * 0.7;
  spheres[2].center.y = cos(global.time * 0.2) * 0.3;

  var col = vec3f(0);
  for(var i=0u; i<samplesPerPixel; i++) {
    let ray = makePrimaryRay(global.width, global.height, 1.0, vec3f(sin(global.time * 0.5), cos(global.time * 0.3) * 0.4, 0.0), vec2f(globalId.xy));
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
