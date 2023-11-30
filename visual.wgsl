struct Global
{
  width: u32,
  height: u32,
  samplesPerPixel: u32,
  maxRecursion: u32,
  rngSeed: f32,
  weight: f32,
  time: f32,
  pad1: f32,
  eye: vec3f,
  vertFov: f32,
  right: vec3f,
  focDist: f32,
  up: vec3f,
  focAngle: f32,
  pixelDeltaX: vec3f,
  pad2: f32,
  pixelDeltaY: vec3f,
  pad3: f32,
  pixelTopLeft: vec3f,
  pad4: f32
}

struct Ray
{
  ori: vec3f,
  dir: vec3f,
  invDir: vec3f
}

struct BvhNode
{
  aabbMin: vec3f,
  startIndex: f32, // Either index of first object or left child node
  aabbMax: vec3f,
  objCount: f32
}

struct Object
{
  shapeType: u32,
  shapeIndex: u32,
  matType: u32,
  matIndex: u32
}

struct Hit
{
  pos: vec3f,
  nrm: vec3f,
  inside: bool,
  matType: u32,
  matIndex: u32
}

const EPSILON = 0.001;
const PI = 3.141592;
const MAX_DIST = 3.402823466e+38;

const SHAPE_TYPE_SPHERE = 0;
const SHAPE_TYPE_PLANE = 1;
const SHAPE_TYPE_BOX = 2;
const SHAPE_TYPE_CYLINDER = 3;
const SHAPE_TYPE_MESH = 4;

const MAT_TYPE_LAMBERT = 0;
const MAT_TYPE_METAL = 1;
const MAT_TYPE_GLASS = 2;

@group(0) @binding(0) var<uniform> globals: Global;
@group(0) @binding(1) var<storage, read> bvhNodes: array<BvhNode>;
@group(0) @binding(2) var<storage, read> objects: array<Object>;
@group(0) @binding(3) var<storage, read> shapes: array<vec4f>;
@group(0) @binding(4) var<storage, read> materials: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> buffer: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> image: array<vec4f>;

var<private> pendingBvhNodeIndices: array<u32, 30>; // Hardcoded size array

var<private> rngState: u32;

// https://jcgt.org/published/0009/03/02/
fn rand() -> f32
{
  rngState = rngState * 747796405u + 2891336453u;
  let word = ((rngState >> ((rngState >> 28u) + 4u)) ^ rngState) * 277803737u;
  return f32((word >> 22u) ^ word) / f32(0xffffffffu);
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

// https://mathworld.wolfram.com/DiskPointPicking.html
fn rand2Disk() -> vec2f
{
  let r = sqrt(rand());
  let theta = 2 * PI * rand();
  return vec2f(r * cos(theta), r * sin(theta));
}

fn minComp(v: vec3f) -> f32
{
  return min(v.x, min(v.y, v.z));
}

fn maxComp(v: vec3f) -> f32
{
  return max(v.x, max(v.y, v.z));
}

fn createRay(ori: vec3f, dir: vec3f) -> Ray
{
  var r: Ray;
  r.ori = ori;
  r.dir = dir;
  r.invDir = 1 / r.dir;
  return r;
}

fn intersectAabb(r: Ray, minDist: f32, maxDist: f32, minExt: vec3f, maxExt: vec3f) -> bool
{
  let t0 = (minExt - r.ori) * r.invDir;
  let t1 = (maxExt - r.ori) * r.invDir;

  let tmin = maxComp(min(t0, t1));
  let tmax = minComp(max(t0, t1));

  return tmin <= tmax && tmax > minDist && tmin < maxDist;
}

fn intersectSphere(r: Ray, tmin: f32, tmax: f32, center: vec3f, radius: f32, dist: ptr<function, f32>) -> bool
{
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

  *dist = t;

  return true;
}

fn completeHitSphere(ray: Ray, dist: f32, center: vec3f, radius: f32, h: ptr<function, Hit>)
{
  (*h).pos = ray.ori + dist * ray.dir;
  (*h).nrm = ((*h).pos - center) / radius;
  (*h).inside = dot(ray.dir, (*h).nrm) > 0;
  (*h).nrm *= select(1.0, -1.0, (*h).inside); 
}

fn evalMaterialLambert(in: Ray, h: Hit, albedo: vec3f, att: ptr<function, vec3f>, outDir: ptr<function, vec3f>) -> bool
{
  let dir = h.nrm + rand3UnitSphere(); 
  *outDir = select(normalize(dir), h.nrm, all(abs(dir) < vec3f(EPSILON)));
  *att = albedo;
  return true;
}

fn evalMaterialMetal(in: Ray, h: Hit, albedo: vec3f, fuzzRadius: f32, att: ptr<function, vec3f>, outDir: ptr<function, vec3f>) -> bool
{
  let dir = reflect(in.dir, h.nrm);
  *outDir = normalize(dir + fuzzRadius * rand3UnitSphere());
  *att = albedo;
  return dot(*outDir, h.nrm) > 0;
}

fn schlickReflectance(cosTheta: f32, refractionIndexRatio: f32) -> f32
{
  var r0 = (1 - refractionIndexRatio) / (1 + refractionIndexRatio);
  r0 = r0 * r0;
  return r0 + (1 - r0) * pow(1 - cosTheta, 5);
}

fn evalMaterialGlass(in: Ray, h: Hit, albedo: vec3f, refractionIndex: f32, att: ptr<function, vec3f>, outDir: ptr<function, vec3f>) -> bool
{
  let refracIndexRatio = select(1 / refractionIndex, refractionIndex, h.inside);
  
  let cosTheta = min(dot(-in.dir, h.nrm), 1);
  /*let sinTheta = sqrt(1 - cosTheta * cosTheta);

  var dir: vec3f;
  if(refracIndexRatio * sinTheta > 1 || schlickReflectance(cosTheta, refracIndexRatio) > rand()) {
    dir = reflect(in.dir, h.nrm);
  } else {
    dir = refract(in.dir, h.nrm, refracIndexRatio);
  }*/

  var dir = refract(in.dir, h.nrm, refracIndexRatio);
  if(all(dir == vec3f(0)) || schlickReflectance(cosTheta, refracIndexRatio) > rand()) {
    dir = reflect(in.dir, h.nrm);
  }

  *outDir = dir;
  *att = albedo;
  return true;
}

fn evalMaterial(in: Ray, h: Hit, att: ptr<function, vec3f>, outDir: ptr<function, vec3f>) -> bool
{
  switch(u32(h.matType))
  {
    case MAT_TYPE_LAMBERT: {
      let data = materials[h.matIndex];
      return evalMaterialLambert(in, h, data.xyz, att, outDir);
    }
    case MAT_TYPE_METAL: {
      let data = materials[h.matIndex];
      return evalMaterialMetal(in, h, data.xyz, data.w, att, outDir);
    }
    case MAT_TYPE_GLASS: {
      let data = materials[h.matIndex];
      return evalMaterialGlass(in, h, data.xyz, data.w, att, outDir);
    }
    default: {
      // TODO Have some error material that outputs red
      return false;
    }
  }
}

fn intersectObjects(ray: Ray, objStartIndex: u32, objCount: u32, tmin: f32, dist: ptr<function, f32>, objId: ptr<function, u32>)
{  
  for(var i=objStartIndex; i<objStartIndex + objCount; i++) {
    let obj = &objects[i];
    switch((*obj).shapeType) {
      case SHAPE_TYPE_SPHERE: {
        let data = shapes[(*obj).shapeIndex];
        var currDist: f32;
        if(intersectSphere(ray, tmin, *dist, data.xyz, data.w, &currDist)) {
          *dist = currDist;
          *objId = i;
        }
      }
      case SHAPE_TYPE_PLANE: {
      }
      default: {
        return;
      }
    }
  }
}

fn intersectScene(ray: Ray, tmin: f32, tmax: f32, hit: ptr<function, Hit>) -> bool
{
  var dist = tmax;
  var objId: u32;

  var pendingPos = 0;
  var pendingCount = 1;

  pendingBvhNodeIndices[0] = 0;

  loop {
    let node = &bvhNodes[pendingBvhNodeIndices[pendingPos]];
    if(intersectAabb(ray, tmin, dist, (*node).aabbMin, (*node).aabbMax)) {
      let nodeObjCount = u32((*node).objCount);
      if(nodeObjCount > 0) {
        intersectObjects(ray, u32((*node).startIndex), nodeObjCount, tmin, &dist, &objId);
      } else {
        let nodeStartIndex = u32((*node).startIndex);
        pendingBvhNodeIndices[pendingPos] = nodeStartIndex;
        pendingBvhNodeIndices[pendingCount] = nodeStartIndex + 1;
        pendingCount += 1;
        continue;
      }
    }
    // TODO Try LIFO/depth first
    // TODO Try L/R node 64 byte aligned
    pendingPos++;
    if(pendingPos >= pendingCount) {
      break;
    }
  }

  // TODO Try without lazy shape data completion

  if(dist < tmax) {
    let obj = &objects[objId];
    switch((*obj).shapeType) {
      case SHAPE_TYPE_SPHERE: {
        let data = shapes[(*obj).shapeIndex];
        completeHitSphere(ray, dist, data.xyz, data.w, hit);
      }
      case SHAPE_TYPE_PLANE: {
      }
      default: {
        return false;
      }
    }

    (*hit).matType = (*obj).matType;
    (*hit).matIndex = (*obj).matIndex;
    
    return true;
  }

  return false;
}

/*fn intersectScene(ray: Ray, tmin: f32, tmax: f32, hit: ptr<function, Hit>) -> bool
{
  var dist = tmax;
  var objId: u32;
 
  intersectObjects(ray, 0u, arrayLength(&objects), tmin, &dist, &objId);

  if(dist < tmax) {
    let obj = &objects[objId];
    switch((*obj).shapeType) {
      case SHAPE_TYPE_SPHERE: {
        let data = shapes[(*obj).shapeIndex];
        completeHitSphere(ray, dist, data.xyz, data.w, hit);
      }
      case SHAPE_TYPE_PLANE: {
      }
      default: {
        return false;
      }
    }

    (*hit).matType = (*obj).matType;
    (*hit).matIndex = (*obj).matIndex;
    
    return true;
  }

  return false;
}*/

fn sampleBackground(ray: Ray) -> vec3f
{
  let t = (ray.dir.y + 1.0) * 0.5;
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
      var newDir: vec3f;
      if(evalMaterial(r, h, &att, &newDir)) {
        col *= att;
        r = createRay(h.pos, newDir);
      } else {
        // Max depth reached or material does not contribute to final color
        break;
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

  return createRay(eyeSample, normalize(pixelSample - eyeSample));
}

@compute @workgroup_size(8,8)
fn computeMain(@builtin(global_invocation_id) globalId: vec3u)
{
  if(all(globalId.xy >= vec2u(globals.width, globals.height))) {
    return;
  }
  
  let index = globals.width * globalId.y + globalId.x;
  rngState = index ^ u32(globals.rngSeed * 0xffffffff); 

  var col = vec3f(0);
  for(var i=0u; i<globals.samplesPerPixel; i++) {
    col += render(createPrimaryRay(vec2f(globalId.xy)), MAX_DIST);
  }

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
