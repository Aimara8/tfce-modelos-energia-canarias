const fs = require("fs");
const path = require("path");

// Construye los datasets finales usados por los modelos.
// Combina los CSV intermedios de demanda, meteorologia municipal y REE.
// Tambien puede consultar REE si se ejecuta con --fetch-ree=true.
const projectRoot = path.resolve(__dirname, "..");
const config = {
  outputDir: "outputs",
  demandMergedFile: "outputs/demand_weather_daily_municipal.csv",
  weatherMunicipalFile: "outputs/weather_daily_municipal_clean.csv",
  reeWideFile: "outputs/ree_renewables_canarias_daily_wide.csv",
  demandOutputFile: "outputs/final_demand_consumption_dataset.csv",
  renewableOutputFile: "outputs/final_renewable_generation_dataset.csv",
};

// Reglas para pasar de meteorologia municipal a meteorologia regional diaria.
// Esta agregacion regional es necesaria para unir clima con generacion renovable.
const weatherAggregationRules = {
  temp_avg_c: "mean",
  temp_max_c: "max",
  temp_min_c: "min",
  pressure_avg_hpa: "mean",
  dew_point_avg_c: "mean",
  precip_intensity_avg_mm: "mean",
  rain_daily_mm: "max",
  humidity_avg_pct: "mean",
  wind_dir_avg_deg: "circular_mean",
  wind_dir_max_deg: "max",
  wind_dir_sdev_deg: "mean",
  wind_speed_avg_ms: "mean",
  wind_speed_max_ms: "max",
  wind_speed_sdev_ms: "mean",
};

// Lee opciones de consola: descarga REE, fecha inicial y fecha final.
function parseArgs(argv) {
  const options = {
    fetchRee: true,
    reeStart: null,
    reeEnd: null,
  };

  for (const arg of argv) {
    if (!arg.startsWith("--")) {
      continue;
    }
    const [rawKey, ...valueParts] = arg.slice(2).split("=");
    const value = valueParts.join("=");
    switch (rawKey.trim()) {
      case "fetch-ree":
        options.fetchRee = value !== "false";
        break;
      case "ree-start":
        options.reeStart = value;
        break;
      case "ree-end":
        options.reeEnd = value;
        break;
      default:
        throw new Error(`Argumento no soportado: --${rawKey}`);
    }
  }

  return options;
}

// Parser CSV con soporte de campos entrecomillados.
function parseCsvLine(line) {
  const fields = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (char === "," && !inQuotes) {
      fields.push(current);
      current = "";
      continue;
    }
    current += char;
  }

  fields.push(current);
  return fields;
}

// Carga un CSV como array de objetos {columna: valor}.
function parseCsvFile(filePath) {
  const resolvedFilePath = path.resolve(projectRoot, filePath);
  const lines = fs.readFileSync(resolvedFilePath, "utf8").split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) {
    return [];
  }
  const headers = parseCsvLine(lines[0]);
  const rows = [];
  for (let i = 1; i < lines.length; i += 1) {
    const fields = parseCsvLine(lines[i]);
    const row = {};
    headers.forEach((header, index) => {
      row[header] = fields[index] ?? "";
    });
    rows.push(row);
  }
  return rows;
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

function parseNumber(value) {
  if (value == null || value === "") {
    return null;
  }
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function round4(value) {
  return value == null ? null : Math.round(value * 10000) / 10000;
}

function getDateRange(rows) {
  let min = null;
  let max = null;
  for (const row of rows) {
    const date = row.date;
    if (!date) {
      continue;
    }
    if (min == null || date < min) {
      min = date;
    }
    if (max == null || date > max) {
      max = date;
    }
  }
  return { min, max };
}

// Crea acumuladores ponderados. El peso suele ser el numero de estaciones
// disponibles, para que municipios con mas soporte meteorologico aporten mas.
function createAggregator(method) {
  if (method === "mean") {
    return { method, weightedSum: 0, weight: 0 };
  }
  if (method === "max" || method === "min") {
    return { method, value: null };
  }
  if (method === "circular_mean") {
    return { method, sinSum: 0, cosSum: 0, weight: 0 };
  }
  throw new Error(`Metodo no soportado: ${method}`);
}

// Anade un valor al acumulador regional.
function updateAggregator(aggregator, value, weight) {
  if (aggregator.method === "mean") {
    aggregator.weightedSum += value * weight;
    aggregator.weight += weight;
    return;
  }
  if (aggregator.method === "max") {
    aggregator.value = aggregator.value == null ? value : Math.max(aggregator.value, value);
    return;
  }
  if (aggregator.method === "min") {
    aggregator.value = aggregator.value == null ? value : Math.min(aggregator.value, value);
    return;
  }
  if (aggregator.method === "circular_mean") {
    const radians = (Math.PI * value) / 180;
    aggregator.sinSum += Math.sin(radians) * weight;
    aggregator.cosSum += Math.cos(radians) * weight;
    aggregator.weight += weight;
  }
}

// Calcula el valor final de cada acumulador.
function finalizeAggregator(aggregator) {
  if (aggregator.method === "mean") {
    return aggregator.weight === 0 ? null : round4(aggregator.weightedSum / aggregator.weight);
  }
  if (aggregator.method === "max" || aggregator.method === "min") {
    return aggregator.value == null ? null : round4(aggregator.value);
  }
  if (aggregator.method === "circular_mean") {
    if (aggregator.weight === 0) {
      return null;
    }
    let angle = (Math.atan2(aggregator.sinSum, aggregator.cosSum) * 180) / Math.PI;
    if (angle < 0) {
      angle += 360;
    }
    return round4(angle);
  }
  throw new Error(`Metodo no soportado: ${aggregator.method}`);
}

// Conserva solo filas de demanda con algun dato meteorologico asociado.
function buildDemandFinalRows(mergedRows) {
  const weatherColumns = Object.keys(weatherAggregationRules);
  return mergedRows.filter((row) =>
    weatherColumns.some((column) => row[column] !== ""),
  );
}

// Agrega la meteorologia municipal por fecha para obtener una serie diaria
// representativa de Canarias completa.
function aggregateCanariasWeather(weatherRows) {
  const rowsByDate = new Map();
  const weatherColumns = Object.keys(weatherAggregationRules);

  for (const row of weatherRows) {
    const date = row.date;
    if (!date) {
      continue;
    }

    let target = rowsByDate.get(date);
    if (!target) {
      target = {
        date,
        municipalitiesWithWeather: new Set(),
        totalStationCount: 0,
        aggregators: Object.create(null),
      };
      rowsByDate.set(date, target);
    }

    const stationWeight = Math.max(parseNumber(row.weather_station_count) || 1, 1);
    let rowHasWeather = false;

    for (const column of weatherColumns) {
      const value = parseNumber(row[column]);
      if (value == null) {
        continue;
      }
      rowHasWeather = true;
      let aggregator = target.aggregators[column];
      if (!aggregator) {
        aggregator = createAggregator(weatherAggregationRules[column]);
        target.aggregators[column] = aggregator;
      }
      updateAggregator(aggregator, value, stationWeight);
    }

    if (rowHasWeather) {
      target.municipalitiesWithWeather.add(row.municipality);
      target.totalStationCount += parseNumber(row.weather_station_count) || 0;
    }
  }

  const finalRows = [];
  for (const row of rowsByDate.values()) {
    const result = {
      date: row.date,
      canarias_weather_municipality_count: row.municipalitiesWithWeather.size,
      canarias_weather_station_count: row.totalStationCount,
    };

    for (const column of weatherColumns) {
      result[column] = row.aggregators[column] ? finalizeAggregator(row.aggregators[column]) : null;
    }

    finalRows.push(result);
  }

  finalRows.sort((a, b) => a.date.localeCompare(b.date));
  return finalRows;
}

function parseDateParts(dateText) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(dateText);
  if (!match) {
    throw new Error(`Fecha invalida: ${dateText}`);
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

function buildYearChunks(startDate, endDate) {
  const startParts = parseDateParts(startDate);
  const endParts = parseDateParts(endDate);
  const start = new Date(Date.UTC(startParts.year, startParts.month - 1, startParts.day));
  const end = new Date(Date.UTC(endParts.year, endParts.month - 1, endParts.day));
  const chunks = [];
  let cursor = new Date(start);

  while (cursor <= end) {
    let chunkEnd = new Date(Date.UTC(cursor.getUTCFullYear(), 11, 31));
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

// Consulta la API de REE para un tramo temporal concreto.
async function fetchReeChunk(startDate, endDate) {
  const baseUrl = "https://apidatos.ree.es/es/datos/generacion/estructura-renovables";
  const url = new URL(baseUrl);
  url.searchParams.set("start_date", `${startDate}T00:00`);
  url.searchParams.set("end_date", `${endDate}T23:59`);
  url.searchParams.set("time_trunc", "day");
  url.searchParams.set("geo_trunc", "electric_system");
  url.searchParams.set("geo_limit", "canarias");
  url.searchParams.set("geo_ids", "8742");

  console.log(`Consultando REE ${startDate} -> ${endDate}`);
  const response = await fetch(url, {
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    throw new Error(`REE devolvio ${response.status} ${response.statusText}`);
  }

  return response.json();
}

// Descarga REE por bloques anuales y transforma la respuesta a filas por fecha.
async function fetchReeRows(startDate, endDate) {
  const byDate = new Map();
  const chunks = buildYearChunks(startDate, endDate);

  for (const chunk of chunks) {
    const payload = await fetchReeChunk(chunk.start, chunk.end);

    for (const entry of payload.included || []) {
      const title = entry?.attributes?.title || entry?.type || "unknown";
      const safeName = title
        .normalize("NFD")
        .replace(/\p{Diacritic}/gu, "")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "");

      for (const valueEntry of entry?.attributes?.values || []) {
        const date = String(valueEntry.datetime || "").slice(0, 10);
        if (!date) {
          continue;
        }
        let row = byDate.get(date);
        if (!row) {
          row = { date };
          byDate.set(date, row);
        }
        row[`ree_${safeName}_value`] = valueEntry.value ?? null;
        row[`ree_${safeName}_pct`] = valueEntry.percentage ?? null;

      }
    }
  }

  const wideRows = Array.from(byDate.values()).sort((a, b) => a.date.localeCompare(b.date));
  return wideRows;
}

// Une datos renovables de REE con meteorologia regional agregada.
function mergeRenewablesAndWeather(reeRows, canariasWeatherRows) {
  const weatherByDate = new Map(canariasWeatherRows.map((row) => [row.date, row]));
  const weatherColumns = [
    "canarias_weather_municipality_count",
    "canarias_weather_station_count",
    ...Object.keys(weatherAggregationRules),
  ];

  return reeRows.map((row) => {
    const weather = weatherByDate.get(row.date);
    const merged = { ...row };
    delete merged.ree_generacion_renovable_pct;
    for (const column of weatherColumns) {
      merged[column] = weather ? weather[column] : null;
    }
    merged.weather_data_source = weather ? "canarias_observations_aggregated" : "";
    return merged;
  });
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const demandRows = parseCsvFile(config.demandMergedFile);
  const weatherRows = parseCsvFile(config.weatherMunicipalFile);
  const demandDateRange = getDateRange(demandRows);

  const demandFinalRows = buildDemandFinalRows(demandRows);
  const canariasWeatherRows = aggregateCanariasWeather(weatherRows);

  let reeWideRows;

  if (options.fetchRee) {
    const reeStart = options.reeStart || demandDateRange.min;
    const reeEnd = options.reeEnd || demandDateRange.max;
    reeWideRows = await fetchReeRows(reeStart, reeEnd);

    const reeWideHeaders = Array.from(
      reeWideRows.reduce((set, row) => {
        Object.keys(row).forEach((key) => set.add(key));
        return set;
      }, new Set(["date"])),
    );
    await writeCsv(config.reeWideFile, reeWideHeaders, reeWideRows);
  } else {
    reeWideRows = parseCsvFile(config.reeWideFile);
  }

  const renewableFinalRows = mergeRenewablesAndWeather(reeWideRows, canariasWeatherRows);

  const demandHeaders = Array.from(
    demandFinalRows.reduce((set, row) => {
      Object.keys(row).forEach((key) => set.add(key));
      return set;
    }, new Set()),
  );
  const renewableHeaders = Array.from(
    renewableFinalRows.reduce((set, row) => {
      Object.keys(row).forEach((key) => set.add(key));
      return set;
    }, new Set(["date"])),
  );

  await writeCsv(config.demandOutputFile, demandHeaders, demandFinalRows);
  await writeCsv(config.renewableOutputFile, renewableHeaders, renewableFinalRows);

  console.log(`Dataset final demanda: ${path.resolve(projectRoot, config.demandOutputFile)}`);
  console.log(`Dataset final renovables: ${path.resolve(projectRoot, config.renewableOutputFile)}`);
  console.log(`Filas demanda final: ${demandFinalRows.length}`);
  console.log(`Municipios demanda final: ${new Set(demandFinalRows.map((row) => row.municipality)).size}`);
  console.log(`Periodo demanda final: ${demandDateRange.min} -> ${demandDateRange.max}`);
  console.log(`Filas renovables final: ${renewableFinalRows.length}`);
  const renewableDateRange = getDateRange(renewableFinalRows);
  console.log(`Periodo renovables final: ${renewableDateRange.min} -> ${renewableDateRange.max}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
