// Frontend del etiquetador. Sin frameworks, solo fetch y DOM nativo.

const formPredict = document.getElementById('form-predict');
const formSearch = document.getElementById('form-search');
const inputImage = document.getElementById('input-image');
const uploadText = document.getElementById('upload-text');

// === Funcionalidad A: subir imagen y predecir forma + color ===

inputImage.addEventListener('change', () => {
    if (inputImage.files.length > 0) {
        uploadText.textContent = inputImage.files[0].name;
    }
});

formPredict.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (inputImage.files.length === 0) {
        alert('Selecciona una imagen primero');
        return;
    }

    const formData = new FormData();
    formData.append('image', inputImage.files[0]);

    const result = document.getElementById('result-predict');
    const resShape = document.getElementById('result-shape');
    const resColors = document.getElementById('result-colors');
    const preview = document.getElementById('preview');

    resShape.textContent = 'analizando...';
    resColors.textContent = '';
    result.classList.remove('oculto');

    // Mostrar preview de la imagen subida
    preview.src = URL.createObjectURL(inputImage.files[0]);

    try {
        const respuesta = await fetch('/predict', {method: 'POST', body: formData});
        const datos = await respuesta.json();

        if (datos.error) {
            resShape.textContent = 'Error: ' + datos.error;
            return;
        }

        resShape.textContent = datos.shape;
        resColors.innerHTML = datos.colors
            .map(c => `<span class="color-chip">${c}</span>`)
            .join('');
    } catch (err) {
        resShape.textContent = 'Error de red: ' + err.message;
    }
});

// === Funcionalidad B: buscar en el catalogo por filtros ===

formSearch.addEventListener('submit', async (e) => {
    e.preventDefault();

    const shape = document.getElementById('select-shape').value;
    const color = document.getElementById('select-color').value;

    const summary = document.getElementById('search-summary');
    const grid = document.getElementById('search-results');

    summary.textContent = 'buscando...';
    summary.classList.remove('oculto');
    grid.innerHTML = '';

    if (!shape && !color) {
        summary.textContent = 'Selecciona al menos un filtro (forma o color).';
        return;
    }

    try {
        const params = new URLSearchParams();
        if (shape) params.append('shape', shape);
        if (color) params.append('color', color);
        params.append('limit', 24);

        const respuesta = await fetch(`/search?${params}`);
        const datos = await respuesta.json();

        summary.textContent = `${datos.total} imagenes encontradas (mostrando ${datos.shown}).`;

        for (const item of datos.results) {
            const div = document.createElement('div');
            div.className = 'item';
            div.innerHTML = `
                <img src="${item.path}" alt="">
                <div class="info">
                    <strong>${item.shape}</strong><br>
                    ${item.colors.join(', ')}
                </div>
            `;
            grid.appendChild(div);
        }
    } catch (err) {
        summary.textContent = 'Error de red: ' + err.message;
    }
});
