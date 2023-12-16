const FULLSCREEN = false;
const ASPECT = 16.0 / 10.0;
const CANVAS_WIDTH = 1280;
const CANVAS_HEIGHT = Math.ceil(CANVAS_WIDTH / ASPECT);

//const ACTIVE_SCENE = "SPHERES";
const ACTIVE_SCENE = "QUADS";
//const ACTIVE_SCENE = "RIOW";

const MAX_RECURSION = 5;
const SAMPLES_PER_PIXEL = 5;
const TEMPORAL_WEIGHT = 0.1;

const BVH_INTERVAL_COUNT = 8;

const MOVE_VELOCITY = 0.1;
const LOOK_VELOCITY = 0.015;

// Size of a bvh node
// aabb min ext, (object/node) start index, aabb max ext, object count
const BVH_NODE_SIZE = 8;

// Size of object data
// shapeType, shapeOfs, matType, matOfs
const OBJECT_SIZE = 4;

// Size of a line of shape data (= vec4f)
const SHAPE_LINE_SIZE = 4;

// Size of a line of material data (= vec4f)
const MAT_LINE_SIZE = 4;

const SHAPE_TYPE_SPHERE = 1;
const SHAPE_TYPE_BOX = 2;
const SHAPE_TYPE_CYLINDER = 3;
const SHAPE_TYPE_QUAD = 4;
const SHAPE_TYPE_MESH = 5;

const MAT_TYPE_LAMBERT = 1;
const MAT_TYPE_METAL = 2;
const MAT_TYPE_GLASS = 3;

const VISUAL_SHADER = `BEGIN_VISUAL_SHADER
END_VISUAL_SHADER`;

let canvas;
let context;
let device;
let globalsBuffer;
let bvhNodesBuffer;
let objectsBuffer;
let shapesBuffer;
let materialsBuffer;
let bindGroup;
let pipelineLayout;
let computePipeline;
let renderPipeline;
let renderPassDescriptor;

let startTime;
let gatheredSamples;

let phi, theta;
let eye, right, up, fwd;
let vertFov, focDist, focAngle;
let pixelDeltaX, pixelDeltaY, pixelTopLeft;

let bvhNodes = [];
let objects = [];
let shapes = [];
let materials = [];

let orbitCam = false;

let rand = xorshift32(471849323);

function loadTextFile(url)
{
  return fetch(url).then(response => response.text());
}

function xorshift32(a)
{
  return function()
  {
    a ^= a << 13;
    a ^= a >>> 17;
    a ^= a << 5;
    return (a >>> 0) / 4294967296;
  }
}

function randRange(min, max)
{
  return min + rand() * (max - min);
}

function vec3Rand()
{
  return[rand(), rand(), rand()];
}

function vec3RandRange(min, max)
{
  return [randRange(min, max), randRange(min, max), randRange(min, max)];
}

function vec3Add(a, b)
{
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function vec3Mul(a, b)
{
  return [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
}

function vec3Negate(v)
{
  return [-v[0], -v[1], -v[2]];
}

function vec3Scale(v, s)
{
  return [v[0] * s, v[1] * s, v[2] * s];
}

function vec3Cross(a, b)
{
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]];
}

function vec3Normalize(v)
{
  let invLen = 1.0 / Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  return [v[0] * invLen, v[1] * invLen, v[2] * invLen];
}

function vec3Length(v)
{
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

function vec3FromSpherical(theta, phi)
{
  return [
    -Math.cos(phi) * Math.sin(theta),
    -Math.cos(theta),
    Math.sin(phi) * Math.sin(theta)];
}

function vec3Min(a, b)
{
  return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.min(a[2], b[2])];
}

function vec3Max(a, b)
{
  return [Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.max(a[2], b[2])];
}

function vec3FromArr(arr, index)
{
  return arr.slice(index, index + 3);
}

function addObject(shapeType, shapeOfs, matType, matOfs)
{
  objects.push(shapeType);
  objects.push(shapeOfs);
  objects.push(matType);
  objects.push(matOfs);
}

function addSphere(center, radius)
{
  shapes.push(...center);
  shapes.push(radius);
  return shapes.length / SHAPE_LINE_SIZE - 1
}

function addQuad(q, u, v)
{
  shapes.push(...q);
  shapes.push(0); // pad
  shapes.push(...u);
  shapes.push(0); // pad
  shapes.push(...v);
  shapes.push(0); // pad
  return shapes.length / SHAPE_LINE_SIZE - 3
}

function addMesh()
{
  // TODO
}

function addLambert(albedo)
{
  materials.push(...albedo);
  materials.push(0); // pad to have full mat line
  return materials.length / MAT_LINE_SIZE - 1;
}

function addMetal(albedo, fuzzRadius)
{
  materials.push(...albedo);
  materials.push(fuzzRadius);
  return materials.length / MAT_LINE_SIZE - 1;
}

function addGlass(albedo, refractionIndex)
{
  materials.push(...albedo);
  materials.push(refractionIndex);
  return materials.length / MAT_LINE_SIZE - 1;
}

function getObjCenter(objIndex)
{
  let objOfs = objIndex * OBJECT_SIZE;
  switch(objects[objOfs]) {
    case SHAPE_TYPE_SPHERE: {
      return vec3FromArr(shapes, objects[objOfs + 1] * SHAPE_LINE_SIZE);
    }
    case SHAPE_TYPE_QUAD: {
      let shapeOfs = objects[objOfs + 1];
      let q = vec3FromArr(shapes, shapeOfs * SHAPE_LINE_SIZE);
      let u = vec3FromArr(shapes, (shapeOfs + 1) * SHAPE_LINE_SIZE);
      let v = vec3FromArr(shapes, (shapeOfs + 2) * SHAPE_LINE_SIZE);
      return vec3Add(vec3Add(q, vec3Scale(u, 0.5)), vec3Scale(v, 0.5));
    }
    default:
      alert("Unknown shape type while retrieving object center");
  }
  return undefined;
}

function getObjAabb(objIndex)
{
  let objOfs = objIndex * OBJECT_SIZE;
  switch(objects[objOfs]) {
    case SHAPE_TYPE_SPHERE: {
      let shapeOfs = objects[objOfs + 1] * SHAPE_LINE_SIZE; 
      let center = vec3FromArr(shapes, shapeOfs);
      let radius = shapes[shapeOfs + 3];
      return { 
        min: vec3Add(center, [-radius, -radius, -radius]),
        max: vec3Add(center, [radius, radius, radius]) };
    }
    case SHAPE_TYPE_QUAD: {
      let shapeOfs = objects[objOfs + 1];
      let q = vec3FromArr(shapes, shapeOfs * SHAPE_LINE_SIZE);
      let u = vec3FromArr(shapes, (shapeOfs + 1) * SHAPE_LINE_SIZE);
      let v = vec3FromArr(shapes, (shapeOfs + 2) * SHAPE_LINE_SIZE);
      let aabb = initAabb();
      growAabb(aabb, q);
      growAabb(aabb, vec3Add(vec3Add(q, u), v));
      padAabb(aabb);
      return aabb;
    }
    default:
      alert("Unknown shape type while retrieving object center");
  }
  return undefined;
}

function initAabb()
{
  return {
    min: [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY],
    max: [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY] };
}

function combineAabbs(a, b)
{
  return { min: vec3Min(a.min, b.min), max: vec3Max(a.max, b.max) };
}

function growAabb(aabb, v)
{
  aabb.min = vec3Min(aabb.min, v);
  aabb.max = vec3Max(aabb.max, v);
}

function padAabb(aabb)
{
  let d = 0.0001;
  for(let i=0; i<3; i++) {
    if(Math.abs(aabb.max[i] - aabb.min[i]) < d) {
      console.log("Padded " + i + ", " + aabb.min[i] + ", " + aabb.max[i]);
      aabb.min[i] -= d * 0.5;
      aabb.max[i] += d * 0.5;
    }
  }
}

function calcAabbArea(aabb)
{
  let d = vec3Add(aabb.max, vec3Negate(aabb.min));
  return d[0] * d[1] + d[1] * d[2] + d[2] * d[0];
}

function calcCenterBounds(objStartIndex, objCount, axis)
{
  let min = Number.POSITIVE_INFINITY, max = Number.NEGATIVE_INFINITY;
  for(let i=0; i<objCount; i++) {
    let center = getObjCenter(objStartIndex + i)[axis];
    min = Math.min(min, center);
    max = Math.max(max, center);
  }
  return { min: min, max: max };
}

function findBestCostIntervalSplit(objStartIndex, objCount, intervalCount)
{
  let bestCost = Number.POSITIVE_INFINITY;
  let bestPos, bestAxis;

  for(let axis=0; axis<3; axis++) {
    // Calculate bounds of object centers
    let bounds = calcCenterBounds(objStartIndex, objCount, axis);
    if(Math.abs(bounds.max - bounds.min) < Number.EPSILON)
      continue;

    // Initialize empty intervals
    let intervals = [];
    for(let i=0; i<intervalCount; i++)
      intervals.push({ aabb: initAabb(), count: 0 });

    // Count objects per interval and find their combined bounds
    let delta = intervalCount / (bounds.max - bounds.min);
    for(let i=0; i<objCount; i++) {
      let center = getObjCenter(objStartIndex + i);
      let intervalIndex = Math.min(intervalCount - 1, (center[axis] - bounds.min) * delta) >>> 0;
      intervals[intervalIndex].aabb = combineAabbs(intervals[intervalIndex].aabb, getObjAabb(objStartIndex + i));
      intervals[intervalIndex].count++;
    }

    // Calculate left/right area and count for each plane separating the intervals
    let areasLeft = new Array(intervalCount - 1);
    let areasRight = new Array(intervalCount - 1);
    let countsLeft = new Array(intervalCount - 1);
    let countsRight = new Array(intervalCount - 1);
    let combinedAabbLeft = initAabb();
    let combinedAabbRight = initAabb();
    let combinedCountLeft = 0;
    let combinedCountRight = 0;
    for(let i=0; i<intervalCount - 1; i++) {
      combinedCountLeft += intervals[i].count;
      countsLeft[i] = combinedCountLeft;
      combinedAabbLeft = combineAabbs(combinedAabbLeft, intervals[i].aabb);
      areasLeft[i] = calcAabbArea(combinedAabbLeft);
      combinedCountRight += intervals[intervalCount - 1 - i].count;
      countsRight[intervalCount - 2 - i] = combinedCountRight;
      combinedAabbRight = combineAabbs(combinedAabbRight, intervals[intervalCount - 1 - i].aabb);
      areasRight[intervalCount - 2 - i] = calcAabbArea(combinedAabbRight);
    }

    // Find best surface area cost of the prepared interval planes
    delta = (bounds.max - bounds.min) / intervalCount;
    for(let i=0; i<intervalCount - 1; i++) {
      let cost = countsLeft[i] * areasLeft[i] + countsRight[i] * areasRight[i];
      if(cost < bestCost) {
        bestCost = cost;
        bestPos = bounds.min + (i + 1) * delta;
        bestAxis = axis;
      }
    }
  }

  return { cost: bestCost, pos: bestPos, axis: bestAxis };
}

function addBvhNode(objStartIndex, objCount)
{
  let nodeAabb = initAabb(); 
  for(let i=0; i<objCount; i++) {
    let objAabb = getObjAabb(objStartIndex + i);
    console.log("obj min: " + objAabb.min + ", max: " + objAabb.max);
    nodeAabb = combineAabbs(nodeAabb, objAabb);
  }

  bvhNodes.push(...nodeAabb.min);
  bvhNodes.push(objStartIndex);
  bvhNodes.push(...nodeAabb.max);
  bvhNodes.push(objCount);

  console.log("node min: " + nodeAabb.min + ", max: " + nodeAabb.max);
  console.log("objStartIndex: " + objStartIndex + ", objCount: " + objCount);
  console.log("---");
}

function subdivideBvhNode(nodeIndex)
{
  let nodeOfs = nodeIndex * BVH_NODE_SIZE;
  let objCount = bvhNodes[nodeOfs + 7];
  let objStartIndex = bvhNodes[nodeOfs + 3];

  // Calc split pos/axis with best cost and compare to no split cost
  let split = findBestCostIntervalSplit(objStartIndex, objCount, BVH_INTERVAL_COUNT);
  let noSplitCost = bvhNodes[nodeOfs + 7] * calcAabbArea({
    min: vec3FromArr(bvhNodes, nodeOfs), max: vec3FromArr(bvhNodes, nodeOfs + 4) });
  if(noSplitCost <= split.cost)
    return;

  // Partition objects to left and right according to split axis/pos
  let l = objStartIndex;
  let r = objStartIndex + objCount - 1;
  while(l <= r) {
    let center = getObjCenter(l);
    if(center[split.axis] < split.pos) {
      l++;
    } else {
      // Swap object data l/r
      let leftObjOfs = l * OBJECT_SIZE;
      let rightObjOfs = r * OBJECT_SIZE;
      for(let i=0; i<OBJECT_SIZE; i++) {
        let t = objects[leftObjOfs + i];
        objects[leftObjOfs + i] = objects[rightObjOfs + i];
        objects[rightObjOfs + i] = t;
      }
      r--;
    }
  }

  // Stop if one side is empty
  let leftObjCount = l - objStartIndex;
  if(leftObjCount == 0 || leftObjCount == objCount)
    return;

  let leftChildIndex = bvhNodes.length / BVH_NODE_SIZE;

  // Not a leaf node, so link left child node
  bvhNodes[nodeOfs + 3] = leftChildIndex;
  bvhNodes[nodeOfs + 7] = 0; // Obj count

  addBvhNode(objStartIndex, leftObjCount);
  addBvhNode(l, objCount - leftObjCount);

  subdivideBvhNode(leftChildIndex);
  subdivideBvhNode(leftChildIndex + 1); // right child
}

function createBvh()
{
  let start = performance.now();
  addBvhNode(0, objects.length / OBJECT_SIZE);
  subdivideBvhNode(0);
  console.log("Create BVH: " + (performance.now() - start).toFixed(3) + " ms, node count: " + bvhNodes.length / BVH_NODE_SIZE);
}

async function createComputePipeline(shaderModule, pipelineLayout, entryPoint)
{
  return device.createComputePipelineAsync({
    layout: pipelineLayout,
    compute: {
      module: shaderModule,
      entryPoint: entryPoint
    }
  });
}

async function createRenderPipeline(shaderModule, pipelineLayout,
  vertexEntry, fragmentEntry)
{
  return device.createRenderPipelineAsync({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: vertexEntry
    },
    fragment: {
      module: shaderModule,
      entryPoint: fragmentEntry,
      targets: [{format: "bgra8unorm"}]
    },
    primitive: {topology: "triangle-strip"}
  });
}

function encodeComputePassAndSubmit(commandEncoder, pipeline, bindGroup,
  countX, countY, countZ)
{
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(countX, countY, countZ);
  passEncoder.end();
}

function encodeRenderPassAndSubmit(commandEncoder, pipeline, bindGroup, view)
{
  renderPassDescriptor.colorAttachments[0].view = view;
  const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.draw(4);
  passEncoder.end();
}

async function createGpuResources(bvhNodesSize, objectsSize,
  shapesSize, materialsSize)
{
  globalsBuffer = device.createBuffer({
    size: 32 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  bvhNodesBuffer = device.createBuffer({
    size: bvhNodesSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });

  objectsBuffer = device.createBuffer({
    size: objectsSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });

  shapesBuffer = device.createBuffer({
    size: shapesSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });

  materialsBuffer = device.createBuffer({
    size: materialsSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });

  let accumulationBuffer = device.createBuffer({
    size: CANVAS_WIDTH * CANVAS_HEIGHT * 4 * 4,
    usage: GPUBufferUsage.STORAGE
  });

  let imageBuffer = device.createBuffer({
    size: CANVAS_WIDTH * CANVAS_HEIGHT * 4 * 4,
    usage: GPUBufferUsage.STORAGE
  });

  let bindGroupLayout = device.createBindGroupLayout({
    entries: [ 
      { binding: 0, 
        visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
        buffer: {type: "uniform"} },
      { binding: 1, 
        visibility: GPUShaderStage.COMPUTE,
        buffer: {type: "read-only-storage"} },
      { binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {type: "read-only-storage"} },
      { binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {type: "read-only-storage"}},
      { binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {type: "read-only-storage"}},
      { binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {type: "storage"}},
      { binding: 6,
        visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
        buffer: {type: "storage"}}
    ]
  });

  bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: globalsBuffer } },
      { binding: 1, resource: { buffer: bvhNodesBuffer } },
      { binding: 2, resource: { buffer: objectsBuffer } },
      { binding: 3, resource: { buffer: shapesBuffer } },
      { binding: 4, resource: { buffer: materialsBuffer } },
      { binding: 5, resource: { buffer: accumulationBuffer } },
      { binding: 6, resource: { buffer: imageBuffer } }
    ]
  });

  pipelineLayout = device.createPipelineLayout(
    { bindGroupLayouts: [bindGroupLayout] } );

  renderPassDescriptor =
    { colorAttachments: [
      { undefined, // view
        clearValue: {r: 1.0, g: 0.0, b: 0.0, a: 1.0},
        loadOp: "clear",
        storeOp: "store"
      } ]
    };

  await createPipelines();
}

async function createPipelines()
{
  let shaderCode;
  if(VISUAL_SHADER.includes("END_VISUAL_SHADER"))
    shaderCode = await loadTextFile("visual.wgsl");
  else
    shaderCode = VISUAL_SHADER;

  let shaderModule = device.createShaderModule({code: shaderCode});

  computePipeline = await createComputePipeline(
    shaderModule, pipelineLayout, "computeMain");
  
  renderPipeline = await createRenderPipeline(
    shaderModule, pipelineLayout, "vertexMain", "fragmentMain");
}

function copySceneData()
{
  device.queue.writeBuffer(bvhNodesBuffer, 0, new Float32Array([...bvhNodes]));
  device.queue.writeBuffer(objectsBuffer, 0, new Uint32Array([...objects]));
  device.queue.writeBuffer(shapesBuffer, 0, new Float32Array([...shapes]));
  device.queue.writeBuffer(materialsBuffer, 0, new Float32Array([...materials]));
}

function copyCanvasData()
{
  device.queue.writeBuffer(globalsBuffer, 0, new Uint32Array([
    CANVAS_WIDTH, CANVAS_HEIGHT, SAMPLES_PER_PIXEL, MAX_RECURSION]));
}

function copyViewData()
{
  device.queue.writeBuffer(globalsBuffer, 8 * 4, new Float32Array([
    ...eye, vertFov,
    ...right, focDist,
    ...up, focAngle,
    ...pixelDeltaX, 0 /* pad */,
    ...pixelDeltaY, 0 /* pad */,
    ...pixelTopLeft, 0 /* pad */]));
}

function copyFrameData(time)
{
  device.queue.writeBuffer(globalsBuffer, 4 * 4, new Float32Array(
    [ rand(),
      SAMPLES_PER_PIXEL / (gatheredSamples + SAMPLES_PER_PIXEL),
      time, 
      /* pad */ 0 ]));
}

let last;

function render(time)
{  
  if(last !== undefined) {
    let frameTime = (performance.now() - last);
    document.title = `${(frameTime).toFixed(2)} / ${(1000.0 / frameTime).toFixed(2)}`;
  }
  last = performance.now();

  if(startTime === undefined)
    startTime = time;

  time = (time - startTime) / 1000.0;
  //setPerformanceTimer();

  update(time);

  copyFrameData(time);
 
  let commandEncoder = device.createCommandEncoder();
  encodeComputePassAndSubmit(commandEncoder, computePipeline, bindGroup,
    Math.ceil(CANVAS_WIDTH / 8), Math.ceil(CANVAS_HEIGHT / 8), 1);
  encodeRenderPassAndSubmit(commandEncoder, renderPipeline, bindGroup,
    context.getCurrentTexture().createView());
  device.queue.submit([commandEncoder.finish()]);

  requestAnimationFrame(render);

  gatheredSamples += SAMPLES_PER_PIXEL;
}

function setPerformanceTimer()
{
  let begin = performance.now();
  device.queue.onSubmittedWorkDone()
    .then(function() {
      let end = performance.now();
      let frameTime = end - begin;
      document.title = `${(frameTime).toFixed(2)} / ${(1000.0 / frameTime).toFixed(2)}`;
    }).catch(function(err) {
      console.log(err);
    });
}

function resetAccumulationBuffer()
{
  gatheredSamples = TEMPORAL_WEIGHT * SAMPLES_PER_PIXEL;
}

function update(time)
{
  if(orbitCam) {
    let speed = 0.3;
    let radius;
    if(ACTIVE_SCENE == "SPHERES")
      radius = 2;
    if(ACTIVE_SCENE == "QUADS")
      radius = 5;
    if(ACTIVE_SCENE == "RIOW")
      radius = 15;
    let height;
    if(ACTIVE_SCENE == "SPHERES" || ACTIVE_SCENE == "QUADS")
      height = 0.5;
    if(ACTIVE_SCENE == "RIOW")
      height = 2.5;
    setView(
      [ Math.sin(time * speed) * radius, height, Math.cos(time * speed) * radius ],
      vec3Normalize(eye));
  }
}

function updateView()
{
  right = vec3Cross([0, 1, 0], fwd);
  up = vec3Cross(fwd, right);

  let viewportHeight = 2 * Math.tan(0.5 * vertFov * Math.PI / 180) * focDist;
  let viewportWidth = viewportHeight * CANVAS_WIDTH / CANVAS_HEIGHT;

  let viewportRight = vec3Scale(right, viewportWidth);
  let viewportDown = vec3Scale(up, -viewportHeight);

  pixelDeltaX = vec3Scale(viewportRight, 1 / CANVAS_WIDTH);
  pixelDeltaY = vec3Scale(viewportDown, 1 / CANVAS_HEIGHT);

  // viewportTopLeft = global.eye - global.focDist * global.fwd - 0.5 * (viewportRight + viewportDown);
  let viewportTopLeft = vec3Add(eye, 
    vec3Add(
      vec3Negate(vec3Scale(fwd, focDist)), 
      vec3Negate(vec3Scale(vec3Add(viewportRight, viewportDown), 0.5))));

  // pixelTopLeft = viewportTopLeft + 0.5 * (v.pixelDeltaX + v.pixelDeltaY)
  pixelTopLeft = vec3Add(viewportTopLeft, 
    vec3Scale(vec3Add(pixelDeltaX, pixelDeltaY), 0.5)); 

  copyViewData();
  resetAccumulationBuffer();
}

function setView(lookFrom, lookAt)
{
  eye = lookFrom;
  fwd = vec3Normalize(vec3Add(lookFrom, vec3Negate(lookAt)));

  theta = Math.acos(-fwd[1]);
  phi = Math.atan2(-fwd[2], fwd[0]) + Math.PI;

  updateView();
}

function resetView()
{
  if(ACTIVE_SCENE == "SPHERES") {
    vertFov = 60;
    focDist = 3;
    focAngle = 0;
    setView([0, 0, 2], [0, 0, 0]);
  }

  if(ACTIVE_SCENE == "QUADS") {
    vertFov = 80;
    focDist = 3;
    focAngle = 0;
    setView([0, 0, 9], [0, 0, 0]);
  }

  if(ACTIVE_SCENE == "RIOW") {
    vertFov = 20;
    focDist = 10;
    focAngle = 0.6;
    setView([13, 2, 3], [0, 0, 0]);
  }
}

function handleKeyEvent(e)
{
  switch (e.key) {
    case "a":
      eye = vec3Add(eye, vec3Scale(right, -MOVE_VELOCITY));
      break;
    case "d":
      eye = vec3Add(eye, vec3Scale(right, MOVE_VELOCITY));
      break;
    case "w":
      eye = vec3Add(eye, vec3Scale(fwd, -MOVE_VELOCITY));
      break;
    case "s":
      eye = vec3Add(eye, vec3Scale(fwd, MOVE_VELOCITY));
      break;
    case ",":
      focDist = Math.max(focDist - 0.1, 0.1);
      break;
    case ".":
      focDist += 0.1;
      break;
    case "-":
      focAngle = Math.max(focAngle - 0.1, 0);
      break;
    case "+":
      focAngle += 0.1;
      break;
    case "r":
      resetView();
      break;
    case "o":
      orbitCam = !orbitCam;
      break;
    case "l":
      createPipelines();
      updateView();
      console.log("Visual shader reloaded");
      break;
  }

  updateView();
}

function handleCameraMouseMoveEvent(e)
{ 
  theta = Math.min(Math.max(theta + e.movementY * LOOK_VELOCITY, 0.01), 0.99 * Math.PI);
  
  phi = (phi - e.movementX * LOOK_VELOCITY) % (2 * Math.PI);
  phi += (phi < 0) ? 2.0 * Math.PI : 0;

  fwd = vec3FromSpherical(theta, phi);

  updateView();
}

async function startRender()
{
  if(FULLSCREEN)
    canvas.requestFullscreen();
  else {
    canvas.style.width = CANVAS_WIDTH;
    canvas.style.height = CANVAS_HEIGHT;
    canvas.style.position = "absolute";
    canvas.style.left = 0;
    canvas.style.top = 0;
  }

  document.querySelector("button").removeEventListener("click", startRender);

  canvas.addEventListener("click", async () => {
    if(!document.pointerLockElement)
      await canvas.requestPointerLock(/*{unadjustedMovement: true}*/);
  });

  document.addEventListener("keydown", handleKeyEvent);

  document.addEventListener("pointerlockchange", () => {
    if(document.pointerLockElement === canvas) {
      canvas.addEventListener("mousemove", handleCameraMouseMoveEvent);
    } else {
      canvas.removeEventListener("mousemove", handleCameraMouseMoveEvent);
    }
  });
  
  requestAnimationFrame(render);
}

function createScene()
{
  if(ACTIVE_SCENE == "SPHERES") {
    addObject(SHAPE_TYPE_SPHERE, addSphere([0, -100.5, 0], 100), MAT_TYPE_LAMBERT, addLambert([0.5, 0.5, 0.5]));
    addObject(SHAPE_TYPE_SPHERE, addSphere([-1, 0, 0], 0.5), MAT_TYPE_LAMBERT, addLambert([0.6, 0.3, 0.3]));

    let glassMatOfs = addGlass([1, 1, 1], 1.5);
    addObject(SHAPE_TYPE_SPHERE, addSphere([0, 0, 0], 0.5), MAT_TYPE_GLASS, glassMatOfs);
    addObject(SHAPE_TYPE_SPHERE, addSphere([0, 0, 0], -0.45), MAT_TYPE_GLASS, glassMatOfs);

    addObject(SHAPE_TYPE_SPHERE, addSphere([1, 0, 0], 0.5), MAT_TYPE_METAL, addMetal([0.3, 0.3, 0.6], 0));
  }

  if(ACTIVE_SCENE == "QUADS") {
    addObject(SHAPE_TYPE_QUAD, addQuad([-3,-2, 5], [0, 0,-4], [0, 4, 0]), MAT_TYPE_LAMBERT, addLambert([1.0, 0.2, 0.2]));
    addObject(SHAPE_TYPE_QUAD, addQuad([-2,-2, 0], [4, 0, 0], [0, 4, 0]), MAT_TYPE_LAMBERT, addLambert([0.2, 1.0, 0.2]));
    addObject(SHAPE_TYPE_QUAD, addQuad([ 3,-2, 1], [0, 0, 4], [0, 4, 0]), MAT_TYPE_LAMBERT, addLambert([0.2, 0.2, 1.0]));
    addObject(SHAPE_TYPE_QUAD, addQuad([-2, 3, 1], [4, 0, 0], [0, 0, 4]), MAT_TYPE_LAMBERT, addLambert([1.0, 0.5, 0.0]));
    addObject(SHAPE_TYPE_QUAD, addQuad([-2,-3, 5], [4, 0, 0], [0, 0,-4]), MAT_TYPE_LAMBERT, addLambert([0.2, 0.8, 0.8]));
    addObject(SHAPE_TYPE_SPHERE, addSphere([0, 0, 2.5], 1.5), MAT_TYPE_GLASS, addGlass([1, 1, 1], 1.5));
    addObject(SHAPE_TYPE_SPHERE, addSphere([0, 0, 2.5], 1.0), MAT_TYPE_LAMBERT, addLambert([0.0, 0.0, 1.0]));
  }

  if(ACTIVE_SCENE == "RIOW") {
    addObject(SHAPE_TYPE_SPHERE, addSphere([0, -1000, 0], 1000), MAT_TYPE_LAMBERT, addLambert([0.5, 0.5, 0.5]));
    addObject(SHAPE_TYPE_SPHERE, addSphere([0, 1, 0], 1), MAT_TYPE_GLASS, addGlass([1, 1, 1], 1.5));
    addObject(SHAPE_TYPE_SPHERE, addSphere([-4, 1, 0], 1), MAT_TYPE_LAMBERT, addLambert([0.4, 0.2, 0.1]));
    addObject(SHAPE_TYPE_SPHERE, addSphere([4, 1, 0], 1), MAT_TYPE_METAL, addMetal([0.7, 0.6, 0.5], 0));

    let SIZE = 11;
    for(a=-SIZE; a<SIZE; a++) {
      for(b=-SIZE; b<SIZE; b++) {
        let matProb = rand();
        let center = [a + 0.9 * rand(), 0.2, b + 0.9 * rand()];
        if(vec3Length(vec3Add(center, [-4, -0.2, 0])) > 0.9) {
          let matType, matOfs;
          if(matProb < 0.8) {
            matType = MAT_TYPE_LAMBERT;
            matOfs = addLambert(vec3Mul(vec3Rand(), vec3Rand()));
          } else if(matProb < 0.95) {
            matType = MAT_TYPE_METAL;
            matOfs = addMetal(vec3RandRange(0.5, 1), randRange(0, 0.5));
          } else {
            matType = MAT_TYPE_GLASS;
            matOfs = addGlass([1, 1, 1], 1.5);
          }
          addObject(SHAPE_TYPE_SPHERE, addSphere(center, 0.2), matType, matOfs);
        }
      }
    }
  }

  console.log("Object count: " + objects.length / OBJECT_SIZE);
}

async function main()
{
  if(!navigator.gpu)
    throw new Error("WebGPU is not supported on this browser.");

  const gpuAdapter = await navigator.gpu.requestAdapter();
  if(!gpuAdapter)
    throw new Error("Can not use WebGPU. No GPU adapter available.");

  device = await gpuAdapter.requestDevice();
  if(!device)
    throw new Error("Failed to request logical device.");

  createScene();
  createBvh();

  await createGpuResources(bvhNodes.length, objects.length, shapes.length, materials.length);
  copyCanvasData();
  copySceneData();

  resetView();
  updateView();

  document.body.innerHTML = "<button>CLICK<canvas style='width:0;cursor:none'>";
  canvas = document.querySelector("canvas");
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;

  let presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  if(presentationFormat !== "bgra8unorm")
    throw new Error(`Expected canvas pixel format of bgra8unorm but was '${presentationFormat}'.`);

  context = canvas.getContext("webgpu");
  context.configure({device, format: presentationFormat, alphaMode: "opaque"});

  if(FULLSCREEN)
    document.querySelector("button").addEventListener("click", startRender);
  else
    startRender();
}

main();
