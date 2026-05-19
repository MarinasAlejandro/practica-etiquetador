"""Tests pequenos y simples para los anadidos recientes.

Cubre:
  - parse_query_text de my_labeling (busqueda textual "pink dress")
  - validacion de /search?limit=xxx en el frontend Flask
  - estabilidad del hash usado en analyze_duplicates

Filosofia: pocos tests, directos, sin dependencias mas alla del propio
proyecto. Se ejecutan con:

    python tests/test_extras.py

Si todos pasan, exit code 0. Si alguno falla, exit code 1 y traza visible.
"""

import os
import sys
import tempfile

# Permitimos correr desde la raiz del proyecto sin pip install -e
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, 'scripts')
for path in (ROOT, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


# -----------------------------------------------------------------------------
# Tests del parser textual
# -----------------------------------------------------------------------------

def test_parse_query_pink_dress():
    """El caso clasico del PDF: 'pink dress' -> color Pink, forma Dresses."""
    import my_labeling as ml
    assert ml.parse_query_text('pink dress') == {'color': 'Pink', 'shape': 'Dresses'}


def test_parse_query_mayusculas():
    """El parser debe ser case-insensitive."""
    import my_labeling as ml
    assert ml.parse_query_text('PINK DRESS') == {'color': 'Pink', 'shape': 'Dresses'}


def test_parse_query_solo_forma():
    """'dress' sin color -> shape Dresses, color None."""
    import my_labeling as ml
    assert ml.parse_query_text('dress') == {'color': None, 'shape': 'Dresses'}


def test_parse_query_flip_flops():
    """Forma de dos palabras: tiene que reconocer 'flip flops'."""
    import my_labeling as ml
    assert ml.parse_query_text('flip flops') == {'color': None, 'shape': 'Flip Flops'}


def test_parse_query_vacio():
    """Texto vacio -> ambos None."""
    import my_labeling as ml
    assert ml.parse_query_text('') == {'color': None, 'shape': None}


def test_parse_query_basura():
    """Tokens no reconocidos no aportan nada."""
    import my_labeling as ml
    assert ml.parse_query_text('lorem ipsum') == {'color': None, 'shape': None}


# -----------------------------------------------------------------------------
# Tests del endpoint /search
# -----------------------------------------------------------------------------

def _make_client():
    """Construye el client de Flask. Carga el dataset (necesario para el app
    al arrancar). Si tarda demasiado en CI, este modulo se puede excluir."""
    # El app.py vive en app/, asi que ajustamos sys.path para importarlo
    # como modulo top-level (importable directamente como 'app').
    app_dir = os.path.join(ROOT, 'app')
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    import app as app_mod  # noqa: WPS433
    return app_mod.app.test_client()


def test_search_limit_no_entero_devuelve_400():
    """limit=abc tiene que dar HTTP 400, no 500."""
    client = _make_client()
    r = client.get('/search?color=Pink&limit=abc')
    assert r.status_code == 400, f"esperado 400, recibido {r.status_code}"
    assert 'limit' in r.get_json().get('error', '').lower()


def test_search_limit_fuera_de_rango_devuelve_400():
    """limit fuera de [1, 200] tiene que dar 400."""
    client = _make_client()
    r = client.get('/search?color=Pink&limit=99999')
    assert r.status_code == 400


def test_search_q_pink_dress_funciona():
    """La busqueda textual del PDF debe devolver resultados de Pink Dresses."""
    client = _make_client()
    r = client.get('/search?q=pink+dress&limit=5')
    assert r.status_code == 200
    data = r.get_json()
    assert data['parsed'] == {'color': 'Pink', 'shape': 'Dresses'}
    # Cada resultado tiene que cumplir AMBOS filtros
    for item in data['results']:
        assert item['shape'] == 'Dresses'
        assert 'Pink' in item['colors']


def test_search_sin_filtros_devuelve_vacio():
    """Sin color, ni forma, ni q -> resultados vacios."""
    client = _make_client()
    r = client.get('/search')
    assert r.status_code == 200
    data = r.get_json()
    assert data['total'] == 0


# -----------------------------------------------------------------------------
# Test del hashing usado para detectar duplicados
# -----------------------------------------------------------------------------

def test_hash_file_estable():
    """Mismo contenido binario -> mismo SHA-256 (sanity check del analisis)."""
    from analyze_duplicates import hash_file
    contenido = b'imagen falsa pero binaria'
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
        f.write(contenido)
        path_a = f.name
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
        f.write(contenido)
        path_b = f.name
    try:
        assert hash_file(path_a) == hash_file(path_b)
    finally:
        os.unlink(path_a)
        os.unlink(path_b)


# -----------------------------------------------------------------------------
# Runner sin pytest
# -----------------------------------------------------------------------------

TESTS = [
    test_parse_query_pink_dress,
    test_parse_query_mayusculas,
    test_parse_query_solo_forma,
    test_parse_query_flip_flops,
    test_parse_query_vacio,
    test_parse_query_basura,
    test_search_limit_no_entero_devuelve_400,
    test_search_limit_fuera_de_rango_devuelve_400,
    test_search_q_pink_dress_funciona,
    test_search_sin_filtros_devuelve_vacio,
    test_hash_file_estable,
]


def main():
    fallos = []
    for fn in TESTS:
        nombre = fn.__name__
        try:
            fn()
            print(f"  OK    {nombre}")
        except AssertionError as e:
            print(f"  FAIL  {nombre}: {e}")
            fallos.append((nombre, str(e)))
        except Exception as e:
            print(f"  ERROR {nombre}: {type(e).__name__}: {e}")
            fallos.append((nombre, repr(e)))

    print()
    print(f"Total: {len(TESTS)} | OK: {len(TESTS) - len(fallos)} | FAIL/ERROR: {len(fallos)}")
    return 1 if fallos else 0


if __name__ == '__main__':
    sys.exit(main())
