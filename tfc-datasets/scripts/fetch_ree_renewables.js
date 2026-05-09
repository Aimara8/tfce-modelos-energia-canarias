const fs = require("fs");
const path = require("path");

const REE_BASE_URL = "https://apidatos.ree.es";
const projectRoot = path.resolve(__dirname, "..");
const DEFAULT_OUTPUT_WIDE = "outputs/ree_renewables_canarias_daily_wide.csv";

function parseArgs(argv) {
  const options = {
    start: "2020-01-01",
    end: "2025-12-31",
    timeTrunc: "day",
    lang: "es",
    geoLimit: "canarias",
    geoTrunc: "electric_system",
    geoId: "8742",
    chunk: "year",
    outputWide: DEFAULT_OUTPUT_WIDE,
  };

  for (const arg of argv) {
    if (!arg.startsWith("--")) {
      continue;
    }
    const [rawKey, ...valueParts] = arg.slice(2).split("=");
    const value = valueParts.join("=");
    const key = rawKey.trim();
    if (!key) {
      continue;
    }

    switch (key) {
      case "start":
        options.start = value;
        break;
      case "end":
        options.end = value;
        break;
      case "time-trunc":
        options.timeTrunc = value;
        break;
      case "lang":
        options.lang = value;
        break;
      case "geo-limit":
        options.geoLimit = value;
        break;
      case "geo-trunc":
        options.geoTrunc = value;
        break;
      case "geo-id":
        options.geoId = value;
        break;
      case "chunk":
        options.chunk = value;
        break;
      case "output-wide":
        options.outputWide = value;
        break;
      default:
        throw new Error(`Argumento no soportado: --${key}`);
    }
  }

  return options;
}

function toIsoStart(dateText) {
  return `${dateText}T00:00`;
}

function toIsoEnd(dateText) {
  return `${dateText}T23:59`;
}

function parseDateParts(dateText) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(dateText);
  if (!match) {
    throw new Error(`Fecha invalida: ${dateText}. Usa YYYY-MM-DD.`);
  }
  return {
    year: Number.parseInt(match[1], 10),
    month: Number.parseInt(match[2], 10),
    day: Number.parseInt(match[3], 10),
  };
}

function formatDateUtc(date) {
  return date.toISOString().slice(0, 10);
}

function buildChunks(startText, endText, mode) {
  const startParts = parseDateParts(startText);
  const endParts = parseDateParts(endText);
  const start = new Date(Date.UTC(startParts.year, startParts.month - 1, startParts.day));
  const end = new Date(Date.UTC(endParts.year, endParts.month - 1, endParts.day));

  if (Number.isNaN(start.valueOf()) || Number.isNaN(end.valueOf()) || start > end) {
    throw new Error("Rango de fechas invalido.");
  }

  if (mode === "none") {
    return [{ start: startText, end: endText }];
  }

  const chunks = [];
  let cursor = new Date(start);

  while (cursor <= end) {
    let chunkEnd;

    if (mode === "month") {
      chunkEnd = new Date(Date.UTC(cursor.getUTCFullYear(), cursor.getUTCMonth() + 1, 0));
    } else if (mode === "year") {
      chunkEnd = new Date(Date.UTC(cursor.getUTCFullYear(), 11, 31));
    } else {
      throw new Error(`Valor de --chunk no soportado: ${mode}`);
    }

    if (chunkEnd > end) {
      chunkEnd = new Date(end);
    }

    chunks.push({
      start: formatDateUtc(cursor),
      end: formatDateUtc(chunkEnd),
    });

    cursor = new Date(chunkEnd);
    cursor.setUTCDate(cursor.getUTCDate() + 1);
  }

  return chunks;
}

function escapeCsv(value) {
  if (value == null) {
    return "";
  }
  const text = String(value);
  if (/[",\r\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

async function writeCsv(filePath, headers, rows) {
  const resolvedFilePath = path.resolve(projectRoot, filePath);
  await fs.promises.mkdir(path.dirname(resolvedFilePath), { recursive: true });
  const stream = fs.createWriteStream(resolvedFilePath, { encoding: "utf8" });
  stream.write(`${headers.join(",")}\n`);
  for (const row of rows) {
    stream.write(`${headers.map((header) => escapeCsv(row[header])).join(",")}\n`);
  }
  await new Promise((resolve, reject) => {
    stream.end(resolve);
    stream.on("error", reject);
  });
}

function slugifyColumnName(text) {
  return text
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

async function fetchChunk(options, chunk) {
  const url = new URL(
    `/${options.lang}/datos/generacion/estructura-renovables`,
    REE_BASE_URL,
  );
  url.searchParams.set("start_date", toIsoStart(chunk.start));
  url.searchParams.set("end_date", toIsoEnd(chunk.end));
  url.searchParams.set("time_trunc", options.timeTrunc);
  url.searchParams.set("geo_trunc", options.geoTrunc);
  url.searchParams.set("geo_limit", options.geoLimit);
  url.searchParams.set("geo_ids", options.geoId);

  console.log(`Consultando REE: ${chunk.start} -> ${chunk.end}`);
  const response = await fetch(url, {
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`REE devolvio ${response.status} ${response.statusText} para ${url.toString()}`);
  }

  const payload = await response.json();
  if (!Array.isArray(payload.included)) {
    throw new Error("La respuesta de REE no contiene 'included'.");
  }
  return payload;
}

function flattenPayload(payload) {
  const rows = [];
  for (const entry of payload.included || []) {
    const attributes = entry.attributes || {};
    const title = attributes.title || entry.type || entry.id || "unknown";

    for (const valueEntry of attributes.values || []) {
      rows.push({
        technology: title,
        datetime: valueEntry.datetime || null,
        date: valueEntry.datetime ? valueEntry.datetime.slice(0, 10) : null,
        value: valueEntry.value ?? null,
        percentage: valueEntry.percentage ?? null,
      });
    }
  }
  return rows;
}

function dedupeRows(rows) {
  const unique = new Map();
  for (const row of rows) {
    const key = [row.technology, row.datetime, row.value, row.percentage].join("|");
    if (!unique.has(key)) {
      unique.set(key, row);
    }
  }
  return Array.from(unique.values()).sort((a, b) => {
    if (a.date !== b.date) {
      return String(a.date).localeCompare(String(b.date));
    }
    return String(a.technology).localeCompare(String(b.technology), "es");
  });
}

function buildWideRows(normalizedRows) {
  const technologies = Array.from(
    new Set(normalizedRows.map((row) => row.technology).filter(Boolean)),
  ).sort((a, b) => a.localeCompare(b, "es"));

  const columns = [];
  const columnMap = new Map();
  for (const technology of technologies) {
    let base = slugifyColumnName(technology);
    if (!base) {
      base = "technology";
    }
    let candidate = base;
    let suffix = 2;
    while (columnMap.has(candidate)) {
      candidate = `${base}_${suffix}`;
      suffix += 1;
    }
    columnMap.set(candidate, technology);
    columns.push(candidate);
  }

  const rowsByDate = new Map();
  for (const row of normalizedRows) {
    if (!row.date) {
      continue;
    }
    let target = rowsByDate.get(row.date);
    if (!target) {
      target = { date: row.date };
      rowsByDate.set(row.date, target);
    }
    const columnName = Array.from(columnMap.entries()).find(([, value]) => value === row.technology)?.[0];
    if (columnName) {
      target[columnName] = row.value;
      target[`${columnName}_pct`] = row.percentage;
    }
  }

  const wideHeaders = ["date"];
  for (const column of columns) {
    wideHeaders.push(column, `${column}_pct`);
  }

  const wideRows = Array.from(rowsByDate.values()).sort((a, b) => a.date.localeCompare(b.date));
  return { wideHeaders, wideRows, technologyColumns: columnMap };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const chunks = buildChunks(options.start, options.end, options.chunk);
  const allRows = [];

  for (const chunk of chunks) {
    const payload = await fetchChunk(options, chunk);
    allRows.push(...flattenPayload(payload));
  }

  const normalizedRows = dedupeRows(allRows);
  const { wideHeaders, wideRows, technologyColumns } = buildWideRows(normalizedRows);

  await writeCsv(options.outputWide, wideHeaders, wideRows);

  console.log(`CSV ancho: ${path.resolve(projectRoot, options.outputWide)}`);
  console.log(`Filas generadas: ${wideRows.length}`);
  console.log(`Tecnologias detectadas: ${technologyColumns.size}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
