from __future__ import annotations

import cv2
import numpy as np

MIN_DESKEW_ANGLE = 0.1
MAX_DESKEW_ANGLE = 20.0
MIN_HEIGHT = 1000


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def estimate_skew(gray: np.ndarray) -> float:
    """Angulo en grados de la inclinacion del texto, via el rectangulo minimo que
    envuelve a los pixeles oscuros."""
    inverted = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90

    # Un angulo grande casi siempre es un rectangulo mal estimado (recibo recortado, fondo
    # con textura), no una foto realmente torcida 40 grados. Enderezar con ese angulo empeora.
    if abs(angle) > MAX_DESKEW_ANGLE:
        return 0.0
    return float(angle)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Non-Local Means. Es el filtro caro de OpenCV (~90 ms) y se probo reemplazarlo por
    `medianBlur`, que cuesta 0,3 ms: el CER medido sobre CORD empeora de 0.462 a 0.495 (7%
    relativo) para ahorrar 26% de latencia. En un OCR, esa no es una buena permuta."""
    return cv2.fastNlMeansDenoising(gray, h=10)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptativo, no Otsu global: un recibo fotografiado tiene iluminacion despareja
    (sombra de la mano, brillo del flash) y un unico umbral global se come media imagen."""
    return cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )


def preprocess(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pipeline canonico para los motores clasicos.

    Devuelve la imagen procesada **y la matriz afin 2x3 que se le aplico**. Sin la matriz, las
    cajas que el OCR encuentra quedan en el espacio de la imagen rotada y reescalada, que no es
    el que ve el usuario: al dibujarlas sobre la foto original aparecerian corridas justo en los
    recibos torcidos, que son los que el deskew existe para arreglar.

    Los VLM comen la imagen original: binarizarla les saca informacion que ellos si aprovechan.
    """
    gray = to_gray(image)
    height, width = gray.shape[:2]

    # El denoise va antes de escalar: sobre la imagen original hay menos pixeles que filtrar y,
    # medido sobre CORD, el CER sale mejor (0.462 vs 0.474) que limpiando despues de agrandar,
    # porque el upscale interpola el ruido en vez de propagarlo ya limpio.
    gray = denoise(gray)

    angle = estimate_skew(gray)
    if abs(angle) < MIN_DESKEW_ANGLE:
        angle = 0.0

    # Tesseract necesita ~30px de alto por caracter; un recibo fotografiado de lejos queda por
    # debajo. Rotacion y escala van en un unico warpAffine: hacerlos por separado resamplea la
    # imagen dos veces, y el segundo pase interpola pixeles que el primero ya habia interpolado.
    scale = max(1.0, MIN_HEIGHT / height)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)

    if angle or scale > 1.0:
        # El centro se corre al escalar: sin esto la imagen agrandada se recorta contra el borde.
        matrix[0, 2] += (width * scale - width) / 2
        matrix[1, 2] += (height * scale - height) / 2
        size = (int(width * scale), int(height * scale))
        gray = cv2.warpAffine(
            gray, matrix, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

    return binarize(gray), matrix


def invert_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Lleva puntos del espacio procesado de vuelta al de la imagen original."""
    inverse = cv2.invertAffineTransform(matrix)
    return cv2.transform(points.reshape(-1, 1, 2).astype(np.float64), inverse).reshape(-1, 2)
