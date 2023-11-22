const FULLSCREEN = false;
const ASPECT = 16.0 / 10.0;
const CANVAS_WIDTH = 1280;
const CANVAS_HEIGHT = Math.ceil(CANVAS_WIDTH / ASPECT);

const ACTIVE_SCENE = "RIOW";
//const ACTIVE_SCENE = "TEST";

const MAX_RECURSION = 10;
const SAMPLES_PER_PIXEL = 1;
const TEMPORAL_WEIGHT = 0;

const MOVE_VELOCITY = 0.05;
const LOOK_VELOCITY = 0.025;

const OBJ_TYPE_SPHERE = 0;
const OBJ_TYPE_PLANE = 1;
const OBJ_TYPE_BOX = 2;
const OBJ_TYPE_CYLINDER = 3;
const OBJ_TYPE_MESH = 4;

const MAT_TYPE_LAMBERT = 0;
const MAT_TYPE_METAL = 1;
const MAT_TYPE_GLASS = 2;

const VISUAL_SHADER = `BEGIN_VISUAL_SHADER
END_VISUAL_SHADER`;

let canvas;
let context;
let device;
let globalsBuffer;
let objectsBuffer;
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

let objects = [];
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
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
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
  // Spherical coordinate axis flipped to accommodate X=right/Y=up
  return [Math.sin(theta) * Math.sin(phi), Math.cos(theta), Math.sin(theta) * Math.cos(phi)];
}

function addSphere(center, radius, materialOffset)
{
  objects.push(OBJ_TYPE_SPHERE);
  objects.push(...center);
  objects.push(radius);
  objects.push(materialOffset);
}

function addPlane(point, normal, materialOffset)
{
  objects.push(OBJ_TYPE_PLANE);
  objects.push(...point);
  objects.push(...normal);
  objects.push(materialOffset);
}

function addLambert(albedo)
{
  materials.push(MAT_TYPE_LAMBERT);
  materials.push(...albedo);
  return materials.length - 4;
}

function addMetal(albedo, fuzzRadius)
{
  materials.push(MAT_TYPE_METAL);
  materials.push(...albedo);
  materials.push(fuzzRadius);
  return materials.length - 5;
}

function addGlass(albedo, refractionIndex)
{
  materials.push(MAT_TYPE_GLASS);
  materials.push(...albedo);
  materials.push(refractionIndex);
  return materials.length - 5;
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

async function createRenderPipeline(shaderModule, pipelineLayout, vertexEntryPoint, fragmentEntryPoint)
{
  return device.createRenderPipelineAsync({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: vertexEntryPoint
    },
    fragment: {
      module: shaderModule,
      entryPoint: fragmentEntryPoint,
      targets: [{format: "bgra8unorm"}]
    },
    primitive: {topology: "triangle-strip"}
  });
}

function encodeComputePassAndSubmit(commandEncoder, pipeline, bindGroup, countX, countY, countZ)
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

async function createGpuResources(objectsSize, materialsSize)
{
  globalsBuffer = device.createBuffer({
    size: 36 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  objectsBuffer = device.createBuffer({
    size: objectsSize * 4,
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
      {binding: 0, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "uniform"}},
      {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}},
      {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}},
      {binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
      {binding: 4, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "storage"}}
    ]
  });

  bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {binding: 0, resource: {buffer: globalsBuffer}},
      {binding: 1, resource: {buffer: objectsBuffer}},
      {binding: 2, resource: {buffer: materialsBuffer}},
      {binding: 3, resource: {buffer: accumulationBuffer}},
      {binding: 4, resource: {buffer: imageBuffer}}
    ]
  });

  pipelineLayout = device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]});

  renderPassDescriptor = {
    colorAttachments: [{
      undefined, // view
      clearValue: {r: 1.0, g: 0.0, b: 0.0, a: 1.0},
      loadOp: "clear",
      storeOp: "store"
    }]
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

  computePipeline = await createComputePipeline(shaderModule, pipelineLayout, "computeMain");
  renderPipeline = await createRenderPipeline(shaderModule, pipelineLayout, "vertexMain", "fragmentMain");
}

function copySceneData()
{
  device.queue.writeBuffer(objectsBuffer, 0, new Float32Array([...objects]));
  device.queue.writeBuffer(materialsBuffer, 0, new Float32Array([...materials]));
}

function copyCanvasData()
{
  device.queue.writeBuffer(globalsBuffer, 0, new Uint32Array([
    CANVAS_WIDTH, CANVAS_HEIGHT, SAMPLES_PER_PIXEL, MAX_RECURSION]));
}

function copyViewData()
{
  device.queue.writeBuffer(globalsBuffer, 12 * 4, new Float32Array([
    ...eye, vertFov,
    ...right, focDist,
    ...up, focAngle,
    ...pixelDeltaX, 0 /* pad */,
    ...pixelDeltaY, 0 /* pad */,
    ...pixelTopLeft, 0 /* pad */]));
}

function copyFrameData(time)
{
  device.queue.writeBuffer(globalsBuffer, 4 * 4, new Float32Array([
    rand(), rand(), rand(), SAMPLES_PER_PIXEL / (gatheredSamples + SAMPLES_PER_PIXEL), time, /* pad */ 0, 0, 0])); 
}

function render(time)
{  
  if(startTime === undefined)
    startTime = time;

  time = (time - startTime) / 1000.0;
  setPerformanceTimer();

  update(time);

  copyFrameData(time);
 
  let commandEncoder = device.createCommandEncoder();
  encodeComputePassAndSubmit(commandEncoder, computePipeline, bindGroup, Math.ceil(CANVAS_WIDTH / 8), Math.ceil(CANVAS_HEIGHT / 8), 1);
  encodeRenderPassAndSubmit(commandEncoder, renderPipeline, bindGroup, context.getCurrentTexture().createView());
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
    let radius = 15;
    let height = 2.5;
    setView([Math.sin(time * speed) * radius, height, Math.cos(time * speed) * radius], vec3Normalize(eye));
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
  let viewportTopLeft = vec3Add(eye, vec3Add(vec3Negate(vec3Scale(fwd, focDist)), vec3Negate(vec3Scale(vec3Add(viewportRight, viewportDown), 0.5))));

  // pixelTopLeft = viewportTopLeft + 0.5 * (v.pixelDeltaX + v.pixelDeltaY)
  pixelTopLeft = vec3Add(viewportTopLeft, vec3Scale(vec3Add(pixelDeltaX, pixelDeltaY), 0.5)); 

  copyViewData();
  resetAccumulationBuffer();
}

function setView(lookFrom, lookAt)
{
  eye = lookFrom;
  fwd = vec3Normalize(vec3Add(lookFrom, vec3Negate(lookAt)));

  theta = Math.acos(fwd[1]);
  phi = Math.acos(fwd[2] / Math.sqrt(fwd[0] * fwd[0] + fwd[2] * fwd[2]));

  updateView();
}

function resetView()
{
  if(ACTIVE_SCENE == "TEST") {
    vertFov = 60;
    focDist = 3;
    focAngle = 0;
    setView([0, 0, 2], [0, 0, 0]);
  }

  if(ACTIVE_SCENE == "RIOW") {
    vertFov = 20;
    focDist = 10;
    focAngle = 0.6;
    setView([13, 2, 3], [0, 0, 0]);
  }
}

function handleCameraKeyEvent(e)
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
  }

  updateView();
}

function handleCameraMouseMoveEvent(e)
{ 
  theta = Math.min(Math.max(theta - e.movementY * LOOK_VELOCITY, 0.01), 0.99 * Math.PI);
  
  phi = (phi - e.movementX * LOOK_VELOCITY) % (2 * Math.PI);
  phi += (phi < 0) ? 2.0 * Math.PI : 0;

  fwd = vec3FromSpherical(theta, phi);

  updateView();
}

async function handleKeyEvent(e)
{    
  switch (e.key) {
    case "l":
      createPipelines();
      updateView();
      console.log("Visual shader reloaded");
      break;
  }
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
      document.addEventListener("keydown", handleCameraKeyEvent);
      canvas.addEventListener("mousemove", handleCameraMouseMoveEvent);
    } else {
      document.removeEventListener("keydown", handleCameraKeyEvent);
      canvas.removeEventListener("mousemove", handleCameraMouseMoveEvent);
    }
  });
  
  requestAnimationFrame(render);
}

function createScene()
{
  if(ACTIVE_SCENE == "TEST") {
    addSphere([0, -100.5, 0], 100, addLambert([0.5, 0.5, 0.5]));
    addSphere([-1, 0, 0], 0.5, addLambert([0.6, 0.3, 0.3]));

    let glassMatOfs = addGlass([1, 1, 1], 1.5);
    addSphere([0, 0, 0], 0.5, glassMatOfs);
    addSphere([0, 0, 0], -0.45, glassMatOfs);

    addSphere([1, 0, 0], 0.5, addMetal([0.3, 0.3, 0.6], 0));
  }

  if(ACTIVE_SCENE == "RIOW") {
    addSphere([0, -1000, 0], 1000, addLambert([0.5, 0.5, 0.5]));
    addSphere([0, 1, 0], 1, addGlass([1, 1, 1], 1.5));
    addSphere([-4, 1, 0], 1, addLambert([0.4, 0.2, 0.1]));
    addSphere([4, 1, 0], 1, addMetal([0.7, 0.6, 0.5], 0));

    for(a=-11; a<11; a++) {
      for(b=-11; b<11; b++) {
        let chooseMat = rand();
        let center = [a + 0.9 * rand(), 0.2, b + 0.9 * rand()];
        if(vec3Length(vec3Add(center, [-4, -0.2, 0])) > 0.9) {
          let mat;
          if(chooseMat < 0.8)
            mat = addLambert(vec3Mul(vec3Rand(), vec3Rand()));
          else if(chooseMat < 0.95)
            mat = addMetal(vec3RandRange(0.5, 1), randRange(0, 0.5));
          else
            mat = addGlass([1, 1, 1], 1.5);
          addSphere(center, 0.2, mat);
        }
      }
    }
  }
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
  await createGpuResources(objects.length, materials.length);

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
