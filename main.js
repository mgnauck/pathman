const FULLSCREEN = false;
const ASPECT = 16.0 / 10.0;
const CANVAS_WIDTH = 1024;
const CANVAS_HEIGHT = Math.ceil(CANVAS_WIDTH / ASPECT);

let canvas;
let context;
let device;
let globalsBuffer;
let sceneBuffer;
let bindGroup;
let pipelineLayout;
let computePipeline;
let renderPipeline;
let renderPassDescriptor;

const MAX_RECURSION = 10;
const SAMPLES_PER_PIXEL = 25;
const TEMPORAL_WEIGHT = 0.1;

let startTime;
let gatheredSamples;

const MOVE_VELOCITY = 0.05;
const LOOK_VELOCITY = 0.025;

let eye, right, up, fwd;
let phi, theta;
let vertFov, focDist, focAngle;

const OBJ_TYPE_PLANE = 1;
const OBJ_TYPE_SPHERE = 2;
const OBJ_TYPE_BOX = 3;
const OBJ_TYPE_CYLINDER = 4;
const OBJ_TYPE_MESH = 5;

const MAT_TYPE_LAMBERT = 1;
const MAT_TYPE_METAL = 2;
const MAT_TYPE_GLASS = 3;

let objectCount = 0;
let objectOffset = 0;
let objectData = [];
let materialOffset = 0;
let materialData = [];

const VISUAL_SHADER = `BEGIN_VISUAL_SHADER
END_VISUAL_SHADER`;

function loadTextFile(url)
{
  return fetch(url).then(response => response.text());
}

function vec3Add(a, b)
{
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
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

function vec3FromSpherical(theta, phi)
{
  // Spherical coordinate axis flipped to accommodate X=right/Y=up
  return [Math.sin(theta) * Math.sin(phi), Math.cos(theta), Math.sin(theta) * Math.cos(phi)];
}

function addSphere(center, radius, materialOffset)
{
  objectData.push(OBJ_TYPE_SPHERE);
  objectData.push(...center);
  objectData.push(radius);
  objectData.push(materialOffset);
  objectCount++;
}

function addPlane(point, normal, materialOffset)
{
  objectData.push(OBJ_TYPE_PLANE);
  objectData.push(...point);
  objectData.push(...normal);
  objectData.push(materialOffset);
  objectCount++;
}

function addMaterial(materialType, albedo)
{  
  materialData.push(materialType);
  materialData.push(...albedo);
}

function addLambert(albedo)
{
  addMaterial(MAT_TYPE_LAMBERT, albedo);
  return materialData.length - 4;
}

function addMetal(albedo, fuzzRadius)
{
  addMaterial(MAT_TYPE_METAL, albedo);
  materialData.push(fuzzRadius);
  return materialData.length - 5;
}

function addGlass(albedo, refractionIndex)
{
  addMaterial(MAT_TYPE_GLASS, albedo);
  materialData.push(refractionIndex);
  return materialData.length - 5;
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

async function createGpuResources(sceneDataSize)
{
  globalsBuffer = device.createBuffer({
    size: 24 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  sceneBuffer = device.createBuffer({
    size: sceneDataSize * 4,
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
      {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
      {binding: 3, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "storage"}}
    ]
  });

  bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {binding: 0, resource: {buffer: globalsBuffer}},
      {binding: 1, resource: {buffer: sceneBuffer}},
      {binding: 2, resource: {buffer: accumulationBuffer}},
      {binding: 3, resource: {buffer: imageBuffer}}
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
  device.queue.writeBuffer(sceneBuffer, 0, new Uint32Array([objectCount, objectData.length]));
  device.queue.writeBuffer(sceneBuffer, 2 * 4, new Float32Array([...objectData]));
  device.queue.writeBuffer(sceneBuffer, (2 + objectData.length) * 4, new Float32Array([...materialData]));
}

function render(time)
{  
  if(startTime === undefined)
    startTime = time;

  time = (time - startTime) / 1000.0;
  setPerformanceTimer();

  update(time);

  device.queue.writeBuffer(globalsBuffer, 0, new Uint32Array([
    CANVAS_WIDTH, CANVAS_HEIGHT, SAMPLES_PER_PIXEL, MAX_RECURSION]));
  device.queue.writeBuffer(globalsBuffer, 16, new Float32Array([
    Math.random(), Math.random(), Math.random(), SAMPLES_PER_PIXEL / (gatheredSamples + SAMPLES_PER_PIXEL),
    ...eye, vertFov,
    ...right, focDist,
    ...up, focAngle,
    ...fwd, time]));
  
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

function update(time)
{
  /*
  let speed = 1.0;
  let radius = 2;
  let height = 0.5;
  eye = [Math.sin(time * speed) * radius, height, Math.cos(time * speed) * radius];
  fwd = vec3Normalize(eye);
  right = vec3Cross([0, 1, 0], fwd);
  up = vec3Cross(fwd, right);

  gatheredSamples = TEMPORAL_WEIGHT * SAMPLES_PER_PIXEL;
  */
}

function calcView()
{
  fwd = vec3FromSpherical(theta, phi);
  right = vec3Cross([0, 1, 0], fwd);
  up = vec3Cross(fwd, right);

  // Resets the accumulation buffer
  gatheredSamples = TEMPORAL_WEIGHT * SAMPLES_PER_PIXEL;
}

function resetView()
{
  vertFov = 60;
  focDist = 3;
  focAngle = 0;

  eye = [0, 0, 2];
  theta = 0.5 * Math.PI;
  phi = 0;
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
  }

  calcView();
}

function handleCameraMouseMoveEvent(e)
{ 
  theta = Math.min(Math.max(theta - e.movementY * LOOK_VELOCITY, 0.01), 0.99 * Math.PI);
  
  phi = (phi - e.movementX * LOOK_VELOCITY) % (2 * Math.PI);
  phi += (phi < 0) ? 2.0 * Math.PI : 0;

  calcView();
}

async function handleKeyEvent(e)
{    
  switch (e.key) {
    case "l":
      createPipelines();
      calcView();
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
  addSphere([0, -100.5, 0], 100, addLambert([0.5, 0.5, 0.5]));
  addSphere([-1, 0, 0], 0.5, addLambert([0.6, 0.3, 0.3]));

  let glassMatOfs = addGlass([1, 1, 1], 1.5);
  addSphere([0, 0, 0], 0.5, glassMatOfs);
  addSphere([0, 0, 0], -0.45, glassMatOfs);

  addSphere([1, 0, 0], 0.5, addMetal([0.3, 0.3, 0.6], 0));
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
  await createGpuResources(2 + objectData.length + materialData.length);
  
  copySceneData();
  
  resetView();
  calcView();

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
