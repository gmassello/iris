import type { Word } from "./types";

interface Props {
  imageUrl: string;
  words: Word[];
  width: number;
  height: number;
}

/**
 * Overlay de bounding boxes en SVG plano, sin libreria de canvas.
 *
 * El truco es el viewBox: al fijarlo a las dimensiones reales de la imagen, las coordenadas
 * que devuelve el motor se dibujan 1:1 sin ninguna cuenta de escalado, y el conjunto sigue
 * siendo responsive porque el SVG se estira con su contenedor.
 *
 * El resaltado va en CSS (`.box:hover`) y no en estado de React: es un efecto puramente visual,
 * y un recibo tiene cientos de palabras. Con estado, cada movimiento del mouse re-renderizaba
 * los cientos de <rect> y volvia a serializar el JSON del recibo.
 *
 * Konva/Fabric existen para *editar* cajas. Estas son de solo lectura.
 */
export function BoxOverlay({ imageUrl, words, width, height }: Props) {
  return (
    <div className="relative w-full">
      <img src={imageUrl} alt="Recibo" className="w-full rounded-lg" />
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="absolute inset-0 h-full w-full"
        preserveAspectRatio="xMidYMid meet"
      >
        {words.map((word, index) => (
          <rect
            key={index}
            x={word.bbox.x}
            y={word.bbox.y}
            width={word.bbox.width}
            height={word.bbox.height}
            className="box"
          >
            <title>{`${word.text} (${(word.confidence * 100).toFixed(0)}%)`}</title>
          </rect>
        ))}
      </svg>
    </div>
  );
}
