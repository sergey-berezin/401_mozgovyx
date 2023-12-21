const orig_canvas = document.getElementById("original_image_canvas");
const api_url = "http://localhost:5041/api/segmentation";

function DrawImage(src, canvas) {
    let ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let img = new Image();
    img.onload = function () {
        let xy_scale = img.width / img.height
        if (xy_scale > 1)
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height / xy_scale);
        else
            ctx.drawImage(img, 0, 0, canvas.width * xy_scale, canvas.height);
    }
    img.src = src;
}

async function GetSegmentation(strImage) {
    let options = {
        method: "POST",
        body: strImage,
        headers: {
            "Content-Type": "application/json"
        }
    };
    let response = await fetch(api_url, options);
    return response.json();
}

function DrawFoundSegmentation(json) {
    let div = document.getElementById("found_segmentation");
    document.getElementById("exception_what").innerText = Object.getOwnPropertyNames(json);
    div.innerHTML = "";
    json.sort(function(a, b) {
        if (a.confidence < b.confidence) return 1;
        return -1;
    });
    for (let i = 0; i < json.length; i++) {
        var object = json[i];
        var obj_div = document.createElement("div");

        var canvas = document.createElement("canvas");
        canvas.height = 100;
        canvas.width = 100;
        obj_div.appendChild(canvas);

        var description_div = document.createElement("span");
        description_div.innerText = object.className + " " + object.confidence;
        obj_div.appendChild(description_div);

        div.appendChild(obj_div);
        DrawImage("data:image/png;base64," + object.image, canvas);
    }
}

async function OnReaderLoad(event) {
    let originalImage = event.target.result;
    try {
        DrawImage(originalImage, orig_canvas);
        document.getElementById("exception_what").innerText = "Processing image";
        let response = await GetSegmentation(JSON.stringify(originalImage.replace("data:image/jpeg;base64,", "")));
        DrawFoundSegmentation(response);
        document.getElementById("exception_what").innerText = "Success";
    }
    catch (exception) {
        document.getElementById("exception_what").innerText = exception;
    }
}

function ListenImageSelector(e) {
    let file = this.files[0];
    if (!file)
        return;
    let reader = new FileReader();
    reader.onload = OnReaderLoad;
    reader.readAsDataURL(file);
}

document.getElementById("image_selector").addEventListener("change", ListenImageSelector);
