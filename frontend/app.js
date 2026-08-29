const MAX_FILE_SIZE = 5 * 1024 * 1024;
const AVAILABLE_CLASSES = [
  { key: 'akiec', label: 'Actinic Keratoses / Intraepithelial Carcinoma' },
  { key: 'bcc', label: 'Basal Cell Carcinoma' },
  { key: 'bkl', label: 'Benign Keratosis-like Lesions' },
  { key: 'df', label: 'Dermatofibroma' },
  { key: 'mel', label: 'Melanoma' },
  { key: 'nv', label: 'Melanocytic Nevi' },
  { key: 'vasc', label: 'Vascular Lesions' }
];

const dropZone = document.getElementById('dropZone');
const imageInput = document.getElementById('imageInput');
const browseButton = document.getElementById('browseButton');
const previewWrapper = document.getElementById('previewWrapper');
const previewImage = document.getElementById('previewImage');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const removeImageButton = document.getElementById('removeImageButton');
const errorBox = document.getElementById('errorBox');
const sampleGallery = document.getElementById('sampleGallery');

const sampleManifest = {
  akiec: [
    '../sample_images/akiec/01.jpg', '../sample_images/akiec/02.jpg', '../sample_images/akiec/03.jpg', '../sample_images/akiec/04.jpg', '../sample_images/akiec/05.jpg', '../sample_images/akiec/06.jpg', '../sample_images/akiec/07.jpg', '../sample_images/akiec/08.jpg', '../sample_images/akiec/09.jpg', '../sample_images/akiec/10.jpg'
  ],
  bcc: [
    '../sample_images/bcc/01.jpg', '../sample_images/bcc/02.jpg', '../sample_images/bcc/03.jpg', '../sample_images/bcc/04.jpg', '../sample_images/bcc/05.jpg', '../sample_images/bcc/06.jpg', '../sample_images/bcc/07.jpg', '../sample_images/bcc/08.jpg', '../sample_images/bcc/09.jpg', '../sample_images/bcc/10.jpg'
  ],
  bkl: [
    '../sample_images/bkl/01.jpg', '../sample_images/bkl/02.jpg', '../sample_images/bkl/03.jpg', '../sample_images/bkl/04.jpg', '../sample_images/bkl/05.jpg', '../sample_images/bkl/06.jpg', '../sample_images/bkl/07.jpg', '../sample_images/bkl/08.jpg', '../sample_images/bkl/09.jpg', '../sample_images/bkl/10.jpg'
  ],
  df: [
    '../sample_images/df/01.jpg', '../sample_images/df/02.jpg', '../sample_images/df/03.jpg', '../sample_images/df/04.jpg', '../sample_images/df/05.jpg', '../sample_images/df/06.jpg', '../sample_images/df/07.jpg', '../sample_images/df/08.jpg', '../sample_images/df/09.jpg', '../sample_images/df/10.jpg'
  ],
  mel: [
    '../sample_images/mel/01.jpg', '../sample_images/mel/02.jpg', '../sample_images/mel/03.jpg', '../sample_images/mel/04.jpg', '../sample_images/mel/05.jpg', '../sample_images/mel/06.jpg', '../sample_images/mel/07.jpg', '../sample_images/mel/08.jpg', '../sample_images/mel/09.jpg', '../sample_images/mel/10.jpg'
  ],
  nv: [
    '../sample_images/nv/01.jpg', '../sample_images/nv/02.jpg', '../sample_images/nv/03.jpg', '../sample_images/nv/04.jpg', '../sample_images/nv/05.jpg', '../sample_images/nv/06.jpg', '../sample_images/nv/07.jpg', '../sample_images/nv/08.jpg', '../sample_images/nv/09.jpg', '../sample_images/nv/10.jpg'
  ],
  vasc: [
    '../sample_images/vasc/01.jpg', '../sample_images/vasc/02.jpg', '../sample_images/vasc/03.jpg', '../sample_images/vasc/04.jpg', '../sample_images/vasc/05.jpg', '../sample_images/vasc/06.jpg', '../sample_images/vasc/07.jpg', '../sample_images/vasc/08.jpg', '../sample_images/vasc/09.jpg', '../sample_images/vasc/10.jpg'
  ]
};

function setError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove('hidden');
}

function clearError() {
  errorBox.textContent = '';
  errorBox.classList.add('hidden');
}

function formatFileSize(size) {
  return size < 1024 ? `${size} B` : `${(size / 1024).toFixed(1)} KB`;
}

function validateFile(file) {
  if (!file) {
    throw new Error('Please select a valid image file.');
  }

  const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
  const extension = file.name.split('.').pop()?.toLowerCase();

  if (!validTypes.includes(file.type) && !['jpg', 'jpeg', 'png'].includes(extension || '')) {
    throw new Error('Unsupported file type. Please upload a JPG, JPEG, or PNG image.');
  }

  if (file.size > MAX_FILE_SIZE) {
    throw new Error('File is too large. Please upload an image smaller than 5 MB.');
  }
}

function handleSelectedFile(file) {
  try {
    validateFile(file);
    clearError();

    const objectUrl = URL.createObjectURL(file);
    previewImage.src = objectUrl;
    previewWrapper.classList.remove('hidden');
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
  } catch (error) {
    setError(error.message);
    removePreview();
  }
}

function removePreview() {
  previewImage.removeAttribute('src');
  previewWrapper.classList.add('hidden');
  fileName.textContent = 'No file selected';
  fileSize.textContent = '0 KB';
  imageInput.value = '';
}

browseButton.addEventListener('click', () => imageInput.click());
removeImageButton.addEventListener('click', () => {
  clearError();
  removePreview();
});

imageInput.addEventListener('change', (event) => {
  const [file] = event.target.files;
  if (file) {
    handleSelectedFile(file);
  }
});

['dragenter', 'dragover'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.remove('dragover');
  });
});

dropZone.addEventListener('drop', (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (file) {
    handleSelectedFile(file);
  }
});

dropZone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    imageInput.click();
  }
});

function renderGallery() {
  sampleGallery.innerHTML = '';

  AVAILABLE_CLASSES.forEach(({ key, label }) => {
    const card = document.createElement('article');
    card.className = 'gallery-card';

    const header = document.createElement('div');
    header.className = 'gallery-header';

    const title = document.createElement('h3');
    title.textContent = label;

    const count = document.createElement('span');
    count.className = 'gallery-count';
    count.textContent = `${sampleManifest[key].length} images`;

    header.append(title, count);

    const grid = document.createElement('div');
    grid.className = 'gallery-image-grid';

    sampleManifest[key].forEach((imagePath) => {
      const image = document.createElement('img');
      image.src = imagePath;
      image.alt = `${label} sample image`;
      image.loading = 'lazy';
      grid.appendChild(image);
    });

    card.append(header, grid);
    sampleGallery.appendChild(card);
  });
}

renderGallery();
