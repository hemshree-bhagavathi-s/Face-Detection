/**
 * Face Detection System
 * Frontend Controller
 * 
 * Webcam -> Base64 Image -> Flask API -> Face Detected / No Face Detected
 */

document.addEventListener("DOMContentLoaded", () => {


    // ===============================
    // UI ELEMENTS
    // ===============================

    const startCamBtn = document.getElementById("startCamBtn");
    const stopCamBtn = document.getElementById("stopCamBtn");

    const webcamVideo = document.getElementById("webcamVideo");
    const frameCanvas = document.getElementById("frameCanvas");

    const webcamPlaceholder = document.getElementById("webcamPlaceholder");

    const detectionBadge = document.getElementById("detectionBadge") || {};
    const badgeIcon = document.getElementById("badgeIcon") || {};
    const badgeText = document.getElementById("badgeText") || {textContent:""};

    const statusDot = document.getElementById("statusDot");
    const liveText = document.getElementById("liveText");


    const frameCountVal = document.getElementById("frameCountVal") || {textContent:""};
    const latencyVal = document.getElementById("latencyVal") || {textContent:""};


    const errorBanner = document.getElementById("errorBanner");
    const errorMessage = document.getElementById("errorMessage");
    const closeErrorBtn = document.getElementById("closeErrorBtn");


    // ===============================
    // VARIABLES
    // ===============================


    let mediaStream = null;

    let captureInterval = null;

    let frameCount = 0;

    let processing = false;


    let apiUrl =
    "https://face-detection-3x3d.onrender.com/predict";


    const canvasContext =
    frameCanvas.getContext("2d");



    // ===============================
    // BUTTON EVENTS
    // ===============================


    startCamBtn.addEventListener(
        "click",
        startCamera
    );


    stopCamBtn.addEventListener(
        "click",
        stopCamera
    );


    closeErrorBtn.addEventListener(
        "click",
        () => {
            errorBanner.classList.add("hidden");
        }
    );


    // ===============================
    // START CAMERA
    // ===============================


    async function startCamera() {


        try {


            startCamBtn.disabled = true;


            mediaStream =
            await navigator.mediaDevices.getUserMedia({

                video: {

                    width: 320,
                    height: 240,
                    facingMode: "user"

                },

                audio:false

            });



            webcamVideo.srcObject =
            mediaStream;


            await webcamVideo.play();



            // Canvas size

            frameCanvas.width = 320;

            frameCanvas.height = 240;



            // Show camera

            webcamPlaceholder.classList.add("hidden");

            webcamVideo.classList.remove("hidden");



            // Live indicator

            statusDot.className =
            "pulse-dot green";


            liveText.textContent =
            "LIVE";



            startCamBtn.disabled =
            true;


            stopCamBtn.disabled =
            false;



            frameCount = 0;

            frameCountVal.textContent =
            "0";



            setStatus(
                "Analyzing..."
            );



            // Start sending frames

            if(captureInterval)
            {
                clearInterval(captureInterval);
            }



            captureInterval =
            setInterval(
                captureFrame,
                1500
            );



        }

        catch(error)
        {

            console.error(
                "Camera Error:",
                error
            );


            showError(
                "Camera permission denied or unavailable"
            );


            startCamBtn.disabled =
            false;

        }


    }





    // ===============================
    // STOP CAMERA
    // ===============================


    function stopCamera(){



        if(captureInterval)
        {

            clearInterval(
                captureInterval
            );

            captureInterval=null;

        }



        if(mediaStream)
        {

            mediaStream
            .getTracks()
            .forEach(track =>
                track.stop()
            );


            mediaStream=null;

        }



        webcamVideo.srcObject=null;



        webcamVideo.classList.add(
            "hidden"
        );


        webcamPlaceholder.classList.remove(
            "hidden"
        );



        statusDot.className =
        "pulse-dot grey";


        liveText.textContent =
        "STANDBY";



        startCamBtn.disabled =
        false;


        stopCamBtn.disabled =
        true;



        setStatus(
            "Awaiting Input"
        );



    }

    // ===============================
    // CAPTURE FRAME AND SEND TO API
    // ===============================


    async function captureFrame(){


        if(!mediaStream || processing)
        {
            return;
        }


        processing = true;


        let startTime =
        performance.now();



        try{


            // Draw video frame on canvas

            canvasContext.drawImage(
                webcamVideo,
                0,
                0,
                frameCanvas.width,
                frameCanvas.height
            );



            // Convert image to Base64

            let imageData =
            frameCanvas.toDataURL(
                "image/jpeg",
                0.5
            );



            // Send image to Flask

            let response =
            await fetch(
                apiUrl,
                {

                    method:"POST",

                    headers:
                    {
                        "Content-Type":
                        "application/json"
                    },


                    body:
                    JSON.stringify(
                    {
                        image:imageData
                    })

                }
            );



            if(!response.ok)
            {

                throw new Error(
                    "Server error"
                );

            }



            let data =
            await response.json();



            let endTime =
            performance.now();



            let latency =
            Math.round(
                endTime-startTime
            );



            latencyVal.textContent =
            latency + " ms";



            frameCount++;


            frameCountVal.textContent =
            frameCount;



            // Update result

            updateDetection(
                data.face_detected
            );



        }


        catch(error)
        {

            console.error(
                "API Error:",
                error
            );


            setStatus(
                "Backend Offline"
            );


        }


        finally
        {

            processing=false;

        }


    }

        // ===============================
    // UPDATE DETECTION RESULT
    // ===============================


    function updateDetection(faceDetected){



        if(faceDetected === true)
        {

            // ==========================
            // FACE DETECTED - GREEN
            // ==========================


            detectionBadge.className =
            "status-badge detected";


            badgeIcon.className =
            "fa-solid fa-circle-check";


            badgeText.textContent =
            "Face Detected";



        }


        else
        {


            // ==========================
            // NO FACE DETECTED - RED
            // ==========================


            detectionBadge.className =
            "status-badge not-detected";


            badgeIcon.className =
            "fa-solid fa-circle-xmark";


            badgeText.textContent =
            "No Face Detected";



        }



    }





    // ===============================
    // SET DEFAULT STATUS
    // ===============================


    function setStatus(message){



        detectionBadge.className =
        "status-badge standby";


        badgeIcon.className =
        "fa-solid fa-pause";


        badgeText.textContent =
        message;



    }

        // ===============================
    // SHOW ERROR MESSAGE
    // ===============================


    function showError(message){


        if(errorBanner && errorMessage)
        {

            errorMessage.textContent =
            message;


            errorBanner.classList.remove(
                "hidden"
            );

        }


    }





    // ===============================
    // HIDE ERROR MESSAGE
    // ===============================


    function hideError(){


        if(errorBanner)
        {

            errorBanner.classList.add(
                "hidden"
            );

        }


    }





    // ===============================
    // CAMERA ERROR HANDLER
    // ===============================


    function handleCameraError(error){


        if(error.name === "NotAllowedError")
        {

            showError(
                "Camera permission denied. Please allow camera access."
            );

        }


        else if(error.name === "NotFoundError")
        {

            showError(
                "No camera device found."
            );

        }


        else
        {

            showError(
                "Unable to access camera."
            );

        }


    }





    // ===============================
    // INTERNET STATUS CHECK
    // ===============================


    window.addEventListener(
        "offline",
        ()=>{

            showError(
                "Internet connection lost."
            );

        }
    );



    window.addEventListener(
        "online",
        ()=>{

            hideError();

        }
    );



});

    // ===============================
    // BUTTON EVENTS
    // ===============================


    startCamBtn.addEventListener(
        "click",
        startCamera
    );


    stopCamBtn.addEventListener(
        "click",
        stopCamera
    );



    // Initial state

    stopCamBtn.disabled = true;


    setStatus(
        "Awaiting Input"
    );