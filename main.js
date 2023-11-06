const FULLSCREEN = false;

const ASPECT = 16.0 / 10.0;
const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = Math.ceil(CANVAS_WIDTH / ASPECT);

let canvas;
let context;
let device;
let uniformBuffer;
let bindGroup;
let pipelineLayout;
let computePipeline;
let renderPipeline;
let renderPassDescriptor;
let startTime;

const MOVE_VELOCITY = 0.05;
const LOOK_VELOCITY = 0.025;

let eye, right, up, fwd;
let vertFov, focDist, focAngle;

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
  return [Math.sin(theta) * Math.cos(phi), Math.sin(theta) * Math.sin(phi), Math.cos(theta)];
}

function vec3Transform(v, m)
{
  const x = v[0];
  const y = v[1];
  const z = v[2];

  return [x * m[0] + y * m[4] + z * m[8],
          x * m[1] + y * m[5] + z * m[9],
          x * m[2] + y * m[6] + z * m[10]];
}

function axisRotation(axis, angle)
{
  let x = axis[0];
  let y = axis[1];
  let z = axis[2];
  const l = 1.0 / Math.sqrt(x * x + y * y + z * z);
  x *= l;
  y *= l;
  z *= l;
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const oneMinusCosine = 1 - c;

  return [xx + (1 - xx) * c, x * y * oneMinusCosine + z * s, x * z * oneMinusCosine - y * s, 0,
          x * y * oneMinusCosine - z * s, yy + (1 - yy) * c, y * z * oneMinusCosine + x * s, 0,
          x * z * oneMinusCosine + y * s, y * z * oneMinusCosine - x * s, zz + (1 - zz) * c, 0,
          0, 0, 0, 1]
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

async function createResources()
{
  uniformBuffer = device.createBuffer({
    // width, height, time, vertFov, focDist, focAngle, eye, right, up, fwd, 6x pad
    size: 24 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  let renderBuffer = device.createBuffer({
    size: CANVAS_WIDTH * CANVAS_HEIGHT * 4 * 4,
    usage: GPUBufferUsage.STORAGE
  });

  let bindGroupLayout = device.createBindGroupLayout({
    entries: [ 
      {binding: 0, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "uniform"}},
      {binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "storage"}}
    ]
  });

  bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {binding: 0, resource: {buffer: uniformBuffer}},
      {binding: 1, resource: {buffer: renderBuffer}}
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

function render(time)
{  
  if(startTime === undefined)
    startTime = time;

  time = (time - startTime) / 1000.0;
  setPerformanceTimer();

  update(time);

  device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([
    CANVAS_WIDTH, CANVAS_HEIGHT, time, vertFov, focDist, focAngle,
    /* pad1 */ 0, /* pad2 */ 0, ...eye, /* pad3 */ 0,
    ...right, /* pad4 */ 0, ...up, /* pad5 */ 0, ...fwd, /* pad6 */ 0]));
  
  let commandEncoder = device.createCommandEncoder();
  encodeComputePassAndSubmit(commandEncoder, computePipeline, bindGroup, Math.ceil(CANVAS_WIDTH / 8), Math.ceil(CANVAS_HEIGHT / 8), 1);
  encodeRenderPassAndSubmit(commandEncoder, renderPipeline, bindGroup, context.getCurrentTexture().createView());
  device.queue.submit([commandEncoder.finish()]);

  requestAnimationFrame(render);
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
  // TODO
}

function setView(e, f)
{
  eye = e;
  fwd = f;
  right = vec3Normalize(vec3Cross([0, 1, 0], fwd));
  up = vec3Cross(fwd, right);

  /*console.log("eye: " + eye);
  console.log("tgt: " + vec3Add(eye, vec3Negate(fwd)));
  console.log("fwd: " + fwd);
  console.log("ri: " + right);
  console.log("up: " + up);*/
}

function resetView()
{
  vertFov = 60;
  focDist = 3;
  focAngle = 0;

  let e = [1, 0.1, 2];
  let t = [0.62, 0.13, 1.1];

  setView(e, vec3Normalize(vec3Add(e, vec3Negate(t))));
}

function handleCameraKeyEvent(e)
{
  switch (e.key) {
    case "a":
      setView(vec3Add(eye, vec3Scale(right, -MOVE_VELOCITY)), fwd);
      break;
    case "d":
      setView(vec3Add(eye, vec3Scale(right, MOVE_VELOCITY)), fwd);
      break;
    case "w":
      setView(vec3Add(eye, vec3Scale(fwd, -MOVE_VELOCITY)), fwd);
      break;
    case "s":
      setView(vec3Add(eye, vec3Scale(fwd, MOVE_VELOCITY)), fwd);
      break;
    case ",":
      focDist = Math.max(focDist - 0.1, 0.1);
      console.log("focDist: " + focDist);
      break;
    case ".":
      focDist += 0.1;
      console.log("focDist: " + focDist);
      break;
    case "-":
      focAngle = Math.max(focAngle - 0.1, 0.0);
      console.log("focAngle: " + focAngle);
      break;
    case "+":
      focAngle += 0.1;
      console.log("focAngle: " + focAngle);
      break;
    case "r":
      resetView();
      break;
  }
}

function handleCameraMouseMoveEvent(e)
{
  let yaw = -e.movementX * LOOK_VELOCITY;
  let pitch = -e.movementY * LOOK_VELOCITY;

  const currentPitch = Math.acos(fwd[1]);
  const newPitch = currentPitch - pitch;
  const minPitch = Math.PI / 180.0;
  const maxPitch = 179.0 * Math.PI / 180.0;

  if(newPitch < minPitch)
    pitch = currentPitch - minPitch;

  if(newPitch > maxPitch)
    pitch = currentPitch - maxPitch;

  setView(eye, vec3Transform(vec3Transform(fwd, axisRotation(right, pitch)), axisRotation([0, 1, 0], yaw)));
}

async function handleKeyEvent(e)
{    
  switch (e.key) {
    case "l":
      createPipelines();
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

  await createResources();
  resetView();

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
