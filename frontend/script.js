const imageInput=document.getElementById("imageInput");

const preview=document.getElementById("preview");

const predictBtn=document.getElementById("predictBtn");

const result=document.getElementById("result");

let selectedFile=null;

imageInput.onchange=()=>{

    selectedFile=imageInput.files[0];

    preview.src=URL.createObjectURL(selectedFile);

    preview.style.display="block";

}

predictBtn.onclick=async()=>{

    if(!selectedFile){

        alert("Choose Image");

        return;

    }

    const formData=new FormData();

    formData.append("image",selectedFile);

    result.innerHTML="Predicting...";

    const response=await fetch(

        "http://127.0.0.1:5000/predict",

        {

            method:"POST",

            body:formData

        }

    );

    const data=await response.json();

    result.innerHTML=

    `
    <h2>${data.prediction}</h2>
    <h3>${data.confidence}%</h3>
    `;

}