import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PMREMGenerator } from 'three';

// =======================
// Global Variables & Constants
// =======================

// Image lists and experiment state
let trialImagesList = [];
let finishTrialImagesList = [];
let annoImagesList = [];
let finishAnnoImagesList = [];
let carImagesList = [];
let carModelsList = [];

let currImage = '';
let currPhrase = 'trial';
let selectedCarType = '';
let userName = '';
let consentDate = '';
let all3Dscenes = {};  // Store 3D scene info keyed by car type

// Car image display order (priority)
const carImageDisplayOrder = [
    "2021_bmw_218i_gran_coupe_15_Family",
    "mini_cooper_s_pre-facelift_City",
    "2022_peugeot_308_Hachback",
    "2014_volkswagen_golf_gti_mk7_Compact",
    "2021_volkswagen_cross_touran_l_280_tsi_MPV",
    "2016_bmw_x5_m_SUV"
];

const carName = {
    "2014_volkswagen_golf_gti_mk7_Compact": "VW Golf",
    "mini_cooper_s_pre-facelift_City": "Mini Cooper",
    "2021_volkswagen_cross_touran_l_280_tsi_MPV": "VW Touran",
    "2021_bmw_218i_gran_coupe_15_Family": "BMW Gran Coupe",
    "2022_peugeot_308_Hachback": "Peugeot 308",
    "2016_bmw_x5_m_SUV": "BMW x5 SUV"
}

// 3D Scene global references (current active scene)
let scene, camera, renderer, controls;
let cubeMesh = null;
let baseCubeSize = null;
let currModelCenter = null;
let currentModel = null;
let selectedIndex = null;

// Debug and deployment flags
const debug = false;

// Initial depth and visual settings
let initDepth = debug ? 0.0 : 0.3;
const desiredSize = 1.5;
const canvasColor = 0x808080;
const cube_color = 0x87CEEB;
const cube_wireframe = false;
let cube_transparent = debug;
let cube_opacity = debug ? 0 : 0;

// Default camera view
const cameraView = { x: 1, y: 0.8, z: 0.8 };
if (debug) {
    cameraView.x = 1.5;
    cameraView.y = 1.2;
    cameraView.z = -1.2; //1.2
} else {
    cameraView.x = 1.0;
    cameraView.y = 0.8;
    cameraView.z = 0.8;
}

// Paths and URLs
const HDRI_PATH = '/assets/textures/autoshop_01_4k.hdr';
const base_url = window.location.origin;

// =======================
// DOM Element References
// =======================
const elements = {
    leftPanel: document.getElementById("left-panel"),
    rightPanel: document.getElementById("right-panel"),
    infoPanel: document.getElementById("info-panel"),
    progressContainer: document.getElementById("progress-container"),
    progressInfo: document.getElementById("progress-info"),
    progressBar: document.getElementById("progress-bar"),
    consentFrame: document.getElementById("consentFrame"),
    instructionStage: document.getElementById("instructionStage"),
    mainInterface: document.getElementById("mainInterface"),
    imageGridContainer: document.getElementById("image-grid-container"),
    allCarImages: document.getElementById("allCarImages"),
    selectCarTypeButton: document.getElementById("selectCarTypeButton"),
    annotateContainer: document.getElementById("annotate-container"),
    carmodelContainer: document.getElementById("carmodel-container"),
    annotateHint: document.getElementById("annotate-hint"),
    nextImageButton: document.getElementById("nextImageButton"),
    changeCarButton: document.getElementById("changeCarType"),
    depthSlider: document.getElementById("depthSlider"),
    depthValue: document.getElementById("depthValue"),
    refImage: document.getElementById("refImage"),
};

// =======================
// Initialization & Event Listeners
// =======================
window.addEventListener("DOMContentLoaded", () => {
    (async () => {


        // Handle messages from iframes or parent windows
        window.addEventListener("message", (event) => {
            switch (event.data.type) {
                case "consent-complete":
                    handlers.handleConsentComplete(event.data);
                    break;
                case "instruction-complete":
                    handlers.handleInstructionComplete();
                    break;
            }
        });

        // UI button listeners
        elements.selectCarTypeButton.addEventListener("click", handlers.handleCarTypeSelect);
        elements.nextImageButton.addEventListener("click", handlers.handleNextImage);
        elements.changeCarButton.addEventListener("click", handlers.handleChangeCar);
        elements.depthSlider.addEventListener("input", handlers.handleDepthSlider);
        elements.depthSlider.value = initDepth;
    })();
});

// =======================
// Data Fetching
// =======================
async function fetchData() {
    try {
        const params = userName ? `?user=${encodeURIComponent(userName)}` : '';
        const [trialRes, annoRes, carImgRes, carModelRes] = await Promise.all([
            fetch(`${base_url}/trial_images/`),
            fetch(`${base_url}/anno_images/${params}`),
            fetch(`${base_url}/car_images/`),
            fetch(`${base_url}/car_models/`),
        ]);

        async function parseImages(res, type) {
            const data = await res.json();
            if (data.data.length !== data.count) {
                throw new Error(`Mismatch in ${type} images count`);
            }
            
            if (type === 'annotation') {
                console.log(`Received ${type} images count: ${data.count}, data length: ${data.data.length}, group:`, data.group);
                
                // Check if there are any images available for annotation
                if (data.count === 0) {
                    console.warn(`No ${type} images available for user: ${userName}`);
                    elements.instructionStage.style.display = "none";
                    elements.mainInterface.classList.remove("hidden");
                    elements.leftPanel.style.display = "none";
                    elements.rightPanel.style.display = "none";
                    elements.progressContainer.style.display = "none";
                    elements.infoPanel.classList.remove("hidden");
                    elements.infoPanel.style.display = "block";
                    elements.infoPanel.innerHTML = `
                        <div style="text-align:center; padding:20px;">
                            <h2 style="font-size:2.5rem;">All Done!</h2>
                            <p style="font-size:1.5rem;">You have already completed all available images for annotation.</p>
                            <p style="font-size:1.2rem; color:#666;">The demo is complete. You may now close this window.</p>
                        </div>
                    `;
                    return [];
                }

            } else {
                console.log(`Received ${type} images count: ${data.count}, data length: ${data.data.length}`);
            }
            return data.data;
        }

        trialImagesList = await parseImages(trialRes, 'trial');
        annoImagesList = await parseImages(annoRes, 'annotation');
        carImagesList = await parseImages(carImgRes, 'car');
        carModelsList = await parseImages(carModelRes, 'car model');

        // Reorder car images based on predefined order
        const carImageMap = carImagesList.reduce((map, path) => {
            const name = path.split("/").pop().split(".")[0];
            map[name] = path;
            return map;
        }, {});

        carImagesList = carImageDisplayOrder.map(name => carImageMap[name]).filter(Boolean);

        // console.log(`Trial images: ${trialImagesList.length}`);
        // console.log(`Annotation images: ${annoImagesList.length}`);
        // console.log(`Car images: ${carImagesList.length}`);
        // console.log(`Car models: ${carModelsList.length}`);

    } catch (err) {
        console.error("Error fetching data:", err);
    }
}

// =======================
// Event Handlers
// =======================
const handlers = {
    handleConsentComplete: async (data) => {
        userName = data.name;
        consentDate = data.date;
        elements.consentFrame.classList.add("hidden");

        try {
            const res = await fetch(`${base_url}/instruction/`);
            const instructions = await res.json();
            document.getElementById('inst-txt').innerHTML = instructions.join('');
        } catch (err) {
            console.error("Failed to fetch instructions:", err);
        }

        elements.instructionStage.style.display = "block";

        const continueBtn = document.getElementById("continueBtn");
        if (continueBtn) {
            continueBtn.addEventListener("click", () => {
                window.parent.postMessage({ type: "instruction-complete" }, "*");
            }, { once: true });
        }

        await fetchData();
        init3DScenesParams();
    },

    handleInstructionComplete: () => {
        elements.instructionStage.style.display = "none";
        elements.mainInterface.classList.remove("hidden");
        elements.imageGridContainer.classList.remove("hidden");
        currPhrase = "trial";
        updateProgress();
        displayNextImage(selectOneImage());
        renderCarImageGrid(carImagesList);
    },

    handleCarTypeSelect: () => {
        if (selectedIndex === null) {
            alert("Please select an image first.");
            return;
        }
        selectedCarType = carImagesList[selectedIndex].split('/').pop().split('.')[0];
        console.log(`Selected car type: ${selectedCarType}`);

        // Prepare UI for annotation
        elements.carmodelContainer.innerHTML = '';
        elements.carmodelContainer.appendChild(elements.annotateHint);

        elements.imageGridContainer.style.display = "none";
        elements.allCarImages.style.display = "none";
        elements.annotateContainer.style.display = "block";
        elements.nextImageButton.style.display = "inline-block";
        elements.carmodelContainer.style.display = "block";
        elements.depthValue.textContent = parseFloat(initDepth).toFixed(2);
        elements.depthSlider.value = initDepth;

        init3DPanel(selectedCarType);
        focusOnSlider();
    },

    handleChangeCar: () => {
        elements.imageGridContainer.style.display = "block";
        elements.allCarImages.style.display = "block";
        elements.annotateContainer.style.display = "none";
        elements.nextImageButton.style.display = "none";
        elements.carmodelContainer.style.display = "none";
        clearImageHighlight();
    },

    handleNextImage: () => {
        finishOneImage();

        const nextImg = selectOneImage();
        if (!nextImg) {
            currPhrase === "trial" ? finishTrialExp() : finishRealExp();
            return;
        }

        elements.imageGridContainer.style.display = "block";
        elements.allCarImages.style.display = "block";
        elements.carmodelContainer.innerHTML = '';
        elements.carmodelContainer.appendChild(elements.annotateHint);
        elements.annotateContainer.style.display = "none";
        elements.depthValue.textContent = parseFloat(initDepth).toFixed(2);
        displayNextImage(nextImg);
    },

    handleDepthSlider: (e) => {
        updateCubeGeometry(parseFloat(e.target.value), currModelCenter);
    },

    handleTryTrialAgain: () => {
        resetUIForTrial();
    },

    handleStartReal: () => {
        resetUIForReal();
    }
};

// =======================
// UI & Experiment Control Functions
// =======================
function focusOnSlider() {
    setInterval(() => {
        if (document.activeElement !== elements.depthSlider) {
            elements.depthSlider.focus();
        }
    }, 500);

    elements.depthSlider.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            handlers.handleNextImage();
        }
    });
}

function selectOneImage() {
    let targetList = currPhrase === "trial" ? trialImagesList : annoImagesList;
    let finishedList = currPhrase === "trial" ? finishTrialImagesList : finishAnnoImagesList;
    const remaining = targetList.filter(img => !finishedList.includes(img));

    if (remaining.length === 0) return null;

    const idx = Math.floor(Math.random() * remaining.length);
    return remaining[idx];
}

function displayNextImage(image) {
    if (!image) throw new Error("No annotation image available");
    currImage = image;
    elements.refImage.src = currImage;
    elements.depthSlider.value = initDepth;
    clearImageHighlight();
}

function finishOneImage() {
    if (currPhrase === "trial") {
        finishTrialImagesList.push(currImage);
    } else {
        finishAnnoImagesList.push(currImage);
    }
    submit_result();
    updateProgress();
}

function submit_result() {
    const senddata = {
        user: userName,
        consentDate: consentDate,
        result: {
            phrase: currPhrase,
            image: currImage,
            car_model: selectedCarType,
            flood_depth: parseFloat(elements.depthSlider.value),
            date_time: new Date().toISOString()
        }
    };

    fetch(`${base_url}/submit_result/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(senddata)
    })
    .then(res => res.json())
    .then(data => console.log('Result saved:', data))
    .catch(console.error);
}

function updateProgress() {
    const isTrial = currPhrase === "trial";
    const total = isTrial ? trialImagesList.length : annoImagesList.length;
    const completed = isTrial ? finishTrialImagesList.length : finishAnnoImagesList.length;

    elements.progressInfo.innerHTML = `
        <div style="text-align:center; margin-bottom:10px;">
            <h3 style="margin-bottom:5px;">
                ${isTrial
                    ? '<span style="color:red; font-weight:bold;">Trial</span> in progress'
                    : '<span style="color:green; font-weight:bold;">Annotation demo</span> in progress'
                }
            </h3>
            <p style="color:#666;">Progress: ${completed} / ${total} images</p>
        </div>
    `;

    elements.progressBar.style.width = `${(completed / total) * 100}%`;
}

function finishTrialExp() {
    elements.leftPanel.style.display = "none";
    elements.rightPanel.style.display = "none";
    elements.progressContainer.style.display = "none";
    elements.infoPanel.classList.remove("hidden");
    elements.infoPanel.style.display = "block";
    elements.infoPanel.innerHTML = `
        <div style="text-align:center; padding:20px;">
            <h2 style="font-size:2.5rem;">Congratulations!</h2>
            <p style="font-size:1.5rem;">You have completed the trial. Please click to start the annotation demo.</p>
            <div class="d-flex justify-content-center gap-3">
                <button id="startRealButton" class="btn btn-success btn-lg">Start the annotation demo</button>
            </div>
        </div>
    `;
    document.getElementById("startRealButton").addEventListener("click", handlers.handleStartReal);
}

function finishRealExp() {
    elements.leftPanel.style.display = "none";
    elements.rightPanel.style.display = "none";
    elements.progressContainer.style.display = "none";
    elements.infoPanel.classList.remove("hidden");
    elements.infoPanel.style.display = "block";
    elements.infoPanel.innerHTML = `
        <div style="text-align:center; padding:20px;">
            <h2 style="font-size:2.5rem;">Thank You!</h2>
            <p style="font-size:1.5rem;">You have completed the demo. Your entries have been saved locally.</p>
            <p style="font-size:1.2rem; color:#666;">You may now close this window.</p>
        </div>
    `;
}

function resetUIForTrial() {
    elements.infoPanel.innerHTML = '';
    elements.infoPanel.classList.add("hidden");
    elements.infoPanel.style.display = "none";
    elements.leftPanel.style.display = "block";
    elements.rightPanel.style.display = "block";
    elements.progressContainer.style.display = "block";

    elements.carmodelContainer.innerHTML = '';
    elements.carmodelContainer.appendChild(elements.annotateHint);

    elements.imageGridContainer.style.display = "block";
    elements.allCarImages.style.display = "block";
    elements.annotateContainer.style.display = "none";

    currPhrase = "trial";
    finishTrialImagesList = [];

    displayNextImage(selectOneImage());
    updateProgress();
}

function resetUIForReal() {
    elements.infoPanel.innerHTML = '';
    elements.infoPanel.classList.add("hidden");
    elements.infoPanel.style.display = "none";
    elements.leftPanel.style.display = "block";
    elements.rightPanel.style.display = "block";
    elements.progressContainer.style.display = "block";

    elements.carmodelContainer.innerHTML = '';
    elements.carmodelContainer.appendChild(elements.annotateHint);

    elements.imageGridContainer.style.display = "block";
    elements.allCarImages.style.display = "block";
    elements.annotateContainer.style.display = "none";

    currPhrase = "real";
    finishAnnoImagesList = [];

    displayNextImage(selectOneImage());
    updateProgress();
}

function clearImageHighlight() {
    const grid = document.getElementById("image-grid");
    Array.from(grid.children).forEach(child => {
        const img = child.querySelector("img.img-thumbnail");
        if (img) {
            img.classList.remove("border", "border-primary");
            img.style.transform = "scale(1)";
            img.style.position = "";
            img.style.zIndex = "";
        }
    });
    selectedIndex = null;
}

// =======================
// Render Car Image Grid
// =======================
function renderCarImageGrid(imageList) {
    const imageGrid = document.getElementById("image-grid");
    imageGrid.innerHTML = '';

    imageList.forEach((imgPath, idx) => {
        const container = document.createElement("div");
        container.style.textAlign = "center";

        const img = document.createElement("img");
        img.src = imgPath;
        img.classList.add("img-thumbnail");
        img.style.width = "100%";
        img.style.cursor = "pointer";

        // Extract simplified car type label
        const imageName = imgPath.split("/").pop().split(".")[0];

        img.addEventListener("click", () => {
            selectedIndex = idx;
            highlightSelectedImage(img);
            elements.selectCarTypeButton.style.display = "inline-block";
        });

        const label = document.createElement("p");
        label.textContent = `${carName[imageName]}`;
        label.style.marginTop = "0px";
        label.style.fontSize = "20px";
        label.style.fontWeight = "bold";
        label.style.color = "black";

        container.appendChild(img);
        container.appendChild(label);
        imageGrid.appendChild(container);
    });
}

function highlightSelectedImage(selectedImg) {
    const imageGrid = document.getElementById("image-grid");
    Array.from(imageGrid.children).forEach(child => {
        const img = child.querySelector("img.img-thumbnail");
        if (img) {
            img.classList.remove("border", "border-primary");
            img.style.transform = "scale(1)";
            img.style.position = "";
            img.style.zIndex = "";
        }
    });

    selectedImg.classList.add("border", "border-primary");
    selectedImg.style.transform = "scale(1.1)";
    selectedImg.style.position = "relative";
    selectedImg.style.zIndex = "10";
}

// =======================
// 3D Scene Initialization and Handling
// =======================

function init3DScenesParams() {
    carImageDisplayOrder.forEach(carType => {
        all3Dscenes[carType] = {
            carType,
            scene: null,
            camera: null,
            renderer: null,
            controls: null,
            cubeMesh: null,
            baseCubeSize: null,
            currModelCenter: null,
            currentModel: null,
            selectedIndex: null,
            currDepth: initDepth,
        };
    });
}

function init3DPanel(selectedCarType) {
    const container = document.getElementById('carmodel-container');

    if (!all3Dscenes[selectedCarType].scene) {
        // Show loading text
        const loadingDiv = document.createElement("div");
        loadingDiv.id = "loading-text";
        loadingDiv.style = "position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:1.5rem;font-weight:bold;color:white;";
        loadingDiv.textContent = "Loading 3D car model ...";
        container.appendChild(loadingDiv);

        // Setup scene
        scene = new THREE.Scene();
        scene.background = new THREE.Color(canvasColor);
        scene.add(new THREE.AmbientLight(canvasColor));

        camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.01, 1000);
        camera.position.set(cameraView.x, cameraView.y, cameraView.z);

        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        container.appendChild(renderer.domElement);

        controls = new OrbitControls(camera, renderer.domElement);
        controls.update();

        // if (debug) {
        //     const axesHelper = new THREE.AxesHelper(2);
        //     scene.add(axesHelper);
        // }

        all3Dscenes[selectedCarType] = {
            carType: selectedCarType,
            scene,
            camera,
            renderer,
            controls,
            cubeMesh: null,
            baseCubeSize: null,
            currModelCenter: null,
            currentModel: null,
            selectedIndex: null,
            currDepth: initDepth
        };

        loadEnvironmentAndModel(selectedCarType);

    } else {
        // Use cached scene
        scene = all3Dscenes[selectedCarType].scene;
        currModelCenter = all3Dscenes[selectedCarType].currModelCenter;
        cubeMesh = all3Dscenes[selectedCarType].cubeMesh;
        baseCubeSize = all3Dscenes[selectedCarType].baseCubeSize;
        updateCubeGeometry(initDepth, currModelCenter);

        camera = all3Dscenes[selectedCarType].camera;
        camera.position.set(cameraView.x, cameraView.y, cameraView.z);

        renderer = all3Dscenes[selectedCarType].renderer;
        controls = all3Dscenes[selectedCarType].controls;

        container.appendChild(renderer.domElement);
    }

    handleResize(selectedCarType);
    animate(selectedCarType);
}

function loadEnvironmentAndModel(selectedCarType) {
    const pmremGenerator = new PMREMGenerator(renderer);
    pmremGenerator.compileEquirectangularShader();

    new RGBELoader()
        .setDataType(THREE.FloatType)
        .load(HDRI_PATH, (hdrTexture) => {
            const envMap = pmremGenerator.fromEquirectangular(hdrTexture).texture;
            scene.environment = envMap;
            hdrTexture.dispose();
            pmremGenerator.dispose();
            loadModel(selectedCarType, envMap);
        });
}

function loadModel(selectedCarType, envMap) {
    if (currentModel) {
        scene.remove(currentModel);
        currentModel.traverse(child => {
            if (child.isMesh) {
                child.geometry.dispose();
                if (child.material) child.material.dispose();
            }
        });
        currentModel = null;
    }

    const loader = new GLTFLoader();
    const modelPath = carModelsList.find(path => path.includes(selectedCarType));

    if (!modelPath) {
        console.error(`Model for car type "${selectedCarType}" not found.`);
        return;
    }

    loader.load(modelPath, (gltf) => {
        currentModel = gltf.scene;
        scene.add(currentModel);

        const box = new THREE.Box3().setFromObject(currentModel);
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = desiredSize / maxDim;
        currentModel.scale.setScalar(scale);

        // if (debug) {
        //     const boxHelper = new THREE.BoxHelper(currentModel);
        //     boxHelper.material.color.set(0xff0000);
        //     scene.add(boxHelper);
        // }

        const boxScaled = new THREE.Box3().setFromObject(currentModel);
        currModelCenter = boxScaled.getCenter(new THREE.Vector3());
        baseCubeSize = size.clone().multiplyScalar(scale);

        currentModel.traverse(child => {
            if (child.isMesh && child.material?.isMeshStandardMaterial) {
                child.material.envMapIntensity = 1.0;
                child.material.needsUpdate = true;
            }
        });

        cubeMesh = addBoundingCube(baseCubeSize, currModelCenter);
        controls.update();

        // Cache scene state
        all3Dscenes[selectedCarType] = {
            ...all3Dscenes[selectedCarType],
            scene,
            camera,
            renderer,
            controls,
            cubeMesh,
            baseCubeSize,
            currModelCenter,
            currentModel
        };

        // Remove loading text
        const loadingText = document.getElementById("loading-text");
        if (loadingText) loadingText.remove();

    }, undefined, error => {
        console.error("Failed to load model:", error);
    });
}

function addBoundingCube(baseCubeSize, currModelCenter) {
    if (cubeMesh) {
        scene.remove(cubeMesh);
        cubeMesh.geometry.dispose();
    }

    const depth = parseFloat(elements.depthSlider.value);
    const geometry = new THREE.BoxGeometry(
        baseCubeSize.x * 1.2,
        baseCubeSize.y * depth,
        baseCubeSize.z * 1.2
    );
    const material = new THREE.MeshBasicMaterial({
        color: cube_color,
        wireframe: cube_wireframe,
        transparent: cube_transparent,
        opacity: cube_opacity
    });

    cubeMesh = new THREE.Mesh(geometry, material);
    geometry.computeBoundingBox();

    cubeMesh.position.set(
        currModelCenter.x,
        currModelCenter.y - (baseCubeSize.y * (1 - depth) / 2),
        currModelCenter.z
    );

    scene.add(cubeMesh);

    return cubeMesh;
}

function updateCubeGeometry(newDepth, currModelCenter) {
    if (!cubeMesh || !baseCubeSize) {
        console.warn("Cube mesh or base cube size not initialized.");
        return;
    }

    const newGeometry = new THREE.BoxGeometry(
        baseCubeSize.x * 1.2,
        baseCubeSize.y * newDepth,
        baseCubeSize.z * 1.2
    );

    cubeMesh.geometry.dispose();
    cubeMesh.geometry = newGeometry;

    newGeometry.computeBoundingBox();

    cubeMesh.position.set(
        currModelCenter.x,
        currModelCenter.y - (baseCubeSize.y * (1 - newDepth) / 2),
        currModelCenter.z
    );
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function handleResize() {
    window.addEventListener('resize', () => {
        const container = document.getElementById('carmodel-container');
        const width = container.clientWidth;
        const height = container.clientHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
    });
}
