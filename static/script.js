const input =
document.getElementById("fileInput");

const preview =
document.getElementById("preview");

input.onchange = evt => {

const [file] = input.files;

if (file) {

preview.src =
URL.createObjectURL(file);

preview.style.display =
"block";

}

};