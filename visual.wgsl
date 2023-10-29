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
  inside: bool
}

struct Sphere
{
  center: vec3f,
  radius: f32
}

const sphere_count = 2;
var<private> spheres = array<Sphere, sphere_count>(
  Sphere(vec3f(0, 0, -1), 0.6),
  Sphere(vec3f(0, -100.5, -1), 100)
);

@group(0) @binding(0) var<uniform> global: Uniforms;
@group(0) @binding(1) var<storage, read_write> buffer: array<vec4f>;

fn rayPos(r: Ray, t: f32) -> vec3f
{
  return r.ori + t * r.dir;
}

fn completeHit(r: Ray, dist: f32, pos: vec3f, nrm: vec3f, h: ptr<function, Hit>)
{
  (*h).t = dist;
  (*h).pos = pos;
  (*h).inside = select(false, true, dot(r.dir, nrm) > 0);
  (*h).nrm = select(nrm, -nrm, (*h).inside);
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

  let sd = sqrt(d);
  var root = (-b - sd) / a;
  if(root <= tmin || tmax <= root) {
    root = (-b + sd) / a;
    if(root <= tmin || tmax <= root) {
      return false;
    }
  }

  let pos = rayPos(r, root);
  completeHit(r, root, pos, (pos - s.center) / s.radius, h);

  return true;
}

fn intersectPrimitives(r: Ray, tmin: f32, tmax: f32, h: ptr<function, Hit>) -> bool
{
  var currMinDist = tmax;
  for(var i=0u; i<sphere_count; i++) {
    var tempHit: Hit;
    if(intersect(spheres[i], r, tmin, currMinDist, &tempHit)) {
      currMinDist = tempHit.t;
      (*h) = tempHit;
    }
  }
  return select(false, true, currMinDist < tmax);
}

fn intersectScene(r: Ray) -> vec3f
{
  var h: Hit;
  if(intersectPrimitives(r, 0, 99999, &h)) {
    return 0.5 * (vec3f(1.0) + h.nrm);
  }

  let t = (normalize(r.dir).y + 1.0) * 0.5;
  return (1.0 - t) * vec3f(1.0) + t * vec3f(0.4, 0.6, 1.0);
}

@compute @workgroup_size(8,8)
fn computeMain(@builtin(global_invocation_id) globalId: vec3u)
{
  if(globalId.x >= u32(global.width) || globalId.y >= u32(global.height)) {
    return;
  }

  let bufMul = vec2u(1, u32(global.width));
  let bufPos = vec2f(globalId.xy);

  let focalLen = 1.0;

  let viewportHeight = 2.0;
  let viewportWidth = viewportHeight * global.width / global.height;

  let viewportRight = vec3f(viewportWidth, 0, 0);
  let viewportDown = vec3f(0, -viewportHeight, 0);

  let pixelDeltaX = viewportRight / global.width;
  let pixelDeltaY = viewportDown / global.height;

  let eye = vec3f(0);

  let viewportTopLeft = eye - vec3f(0, 0, focalLen) - viewportRight * 0.5 - viewportDown * 0.5;
  let pixelTopLeft = viewportTopLeft + (pixelDeltaX + pixelDeltaY) * 0.5;

  let pixelTarget = pixelTopLeft + pixelDeltaX * bufPos.x + pixelDeltaY * bufPos.y;
  let dir = pixelTarget - eye;

  let ray = Ray(eye, dir);

  let col = intersectScene(ray);

  buffer[dot(globalId.xy, bufMul)] = vec4f(col, 1.0);
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
