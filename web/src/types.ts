export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Word {
  text: string;
  bbox: BoundingBox;
  confidence: number;
}

export interface LineItem {
  description: string;
  quantity: number | null;
  unit_price: number | null;
  total: number | null;
}

export interface Receipt {
  merchant: string | null;
  tax_id: string | null;
  date: string | null;
  items: LineItem[];
  subtotal: number | null;
  tax: number | null;
  total: number | null;
  currency: string | null;
}

export interface OCRResult {
  engine: string;
  text: string;
  words: Word[];
  image_width: number;
  image_height: number;
  elapsed_ms: number;
  receipt: Receipt | null;
}
