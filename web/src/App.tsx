import { useEffect, useState } from "react";
import { BoxOverlay } from "./BoxOverlay";
import type { OCRResult } from "./types";

const money = (value: number | null) =>
  value === null ? <span className="text-slate-400">—</span> : value.toLocaleString("en-US");

export default function App() {
  const [engines, setEngines] = useState<string[]>([]);
  const [engine, setEngine] = useState("tesseract");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [result, setResult] = useState<OCRResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((response) => response.json())
      .then((data) => setEngines(data.engines))
      .catch(() => setError("could not reach the API"));
  }, []);

  async function upload(file: File) {
    setLoading(true);
    setError(null);
    setResult(null);
    setImageUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return URL.createObjectURL(file);
    });

    const body = new FormData();
    body.append("file", file);

    try {
      const response = await fetch(`/api/extract?engine=${engine}`, { method: "POST", body });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(detail.detail ?? "extraction failed");
      }
      setResult(await response.json());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "unknown error");
    } finally {
      setLoading(false);
    }
  }

  const receipt = result?.receipt;

  return (
    <div className="min-h-screen bg-slate-50 p-8 text-slate-900">
      <header className="mx-auto mb-8 max-w-6xl">
        <h1 className="text-3xl font-semibold tracking-tight">iris</h1>
        <p className="text-slate-600">
          Receipt OCR with interchangeable engines. Hover a box to highlight the word.
        </p>
      </header>

      <div className="mx-auto max-w-6xl">
        <div className="mb-6 flex flex-wrap items-center gap-4 rounded-lg border border-slate-200 bg-white p-4">
          <label className="flex items-center gap-2">
            <span className="text-sm font-medium">Engine</span>
            <select
              value={engine}
              onChange={(event) => setEngine(event.target.value)}
              className="rounded border border-slate-300 px-3 py-1.5 text-sm"
            >
              {engines.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>

          <input
            type="file"
            accept="image/*"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void upload(file);
            }}
            className="text-sm file:mr-3 file:rounded file:border-0 file:bg-slate-900 file:px-4 file:py-1.5 file:text-white"
          />

          {loading && <span className="text-sm text-slate-500">processing…</span>}
          {result && (
            <span className="ml-auto text-sm text-slate-500">
              {result.engine} · {result.elapsed_ms.toFixed(0)} ms · {result.words.length} words
            </span>
          )}
        </div>

        {error && (
          <div className="mb-6 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
            {error}
          </div>
        )}

        <div className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-lg border border-slate-200 bg-white p-4">
            {imageUrl && result ? (
              <BoxOverlay
                imageUrl={imageUrl}
                words={result.words}
                width={result.image_width}
                height={result.image_height}
              />
            ) : imageUrl ? (
              <img src={imageUrl} alt="Receipt" className="w-full rounded-lg" />
            ) : (
              <p className="py-24 text-center text-slate-400">Upload a photo of a receipt</p>
            )}

            {result && result.words.length === 0 && (
              <p className="mt-3 text-sm text-slate-500">
                <strong>{result.engine}</strong> returns the receipt already structured and emits no
                bounding boxes, so there is nothing to draw.
              </p>
            )}
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-4">
            {receipt ? (
              <>
                <dl className="mb-4 grid grid-cols-2 gap-2 text-sm">
                  <dt className="text-slate-500">Merchant</dt>
                  <dd className="font-medium">{receipt.merchant ?? "—"}</dd>
                  <dt className="text-slate-500">Tax ID</dt>
                  <dd className="font-medium">{receipt.tax_id ?? "—"}</dd>
                  <dt className="text-slate-500">Date</dt>
                  <dd className="font-medium">{receipt.date ?? "—"}</dd>
                </dl>

                <table className="mb-4 w-full text-sm">
                  <thead className="border-b border-slate-200 text-left text-slate-500">
                    <tr>
                      <th className="pb-1 font-medium">Item</th>
                      <th className="pb-1 text-right font-medium">Amount</th>
                    </tr>
                  </thead>
                  <tbody>
                    {receipt.items.map((item, index) => (
                      <tr key={index} className="border-b border-slate-100">
                        <td className="py-1">{item.description}</td>
                        <td className="py-1 text-right tabular-nums">{money(item.total)}</td>
                      </tr>
                    ))}
                  </tbody>
                  <tfoot className="font-medium">
                    <tr>
                      <td className="pt-2">Subtotal</td>
                      <td className="pt-2 text-right tabular-nums">{money(receipt.subtotal)}</td>
                    </tr>
                    <tr>
                      <td>Tax</td>
                      <td className="text-right tabular-nums">{money(receipt.tax)}</td>
                    </tr>
                    <tr className="text-base">
                      <td className="pt-1">Total</td>
                      <td className="pt-1 text-right tabular-nums">{money(receipt.total)}</td>
                    </tr>
                  </tfoot>
                </table>

                <details>
                  <summary className="cursor-pointer text-sm text-slate-500">Raw JSON</summary>
                  <pre className="mt-2 overflow-x-auto rounded bg-slate-900 p-3 text-xs text-slate-100">
                    {JSON.stringify(receipt, null, 2)}
                  </pre>
                </details>
              </>
            ) : (
              <p className="py-24 text-center text-slate-400">The structured receipt shows up here</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
