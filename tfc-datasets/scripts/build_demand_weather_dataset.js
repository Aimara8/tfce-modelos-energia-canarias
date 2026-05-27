const fs = require("fs");
const path = require("path");
const readline = require("readline");

// Construye el dataset base de consumo + meteorologia observada.
// Entradas: ISTAC para demanda electrica y SITCAN para estaciones/observaciones.
// Salidas: estaciones limpias, consumo limpio, meteorologia diaria municipal
// y dataset unido consumo + meteorologia.
const projectRoot = path.resolve(__dirname, "..");
const config = {
  basePath: projectRoot,
  istacFile: "inputs/istac/dataset-ISTAC_C00022A_000005_1.11_20260426202346.csv",
  legacyIstacFile: "dataset-ISTAC_C00022A_000005_1.11_20260426202346.csv",
  stationsFile: "inputs/weather/estaciones.csv",
  legacyStationsFile: "estaciones.csv",
  weatherDir: "inputs/weather",
  weatherPattern: /^observaciones_\d{4}\.csv$/i,
  outputDir: "outputs",
};

// Corrige variantes de nombres municipales para no duplicar entidades.
const municipalityAliases = new Map([
  ["Fuencaliente", "Fuencaliente de La Palma"],
  ["Guia", "Santa María de Guía de Gran Canaria"],
  ["Mazo", "Villa de Mazo"],
  ["Santa Maria de Guia", "Santa María de Guía de Gran Canaria"],
  ["Vilaflor", "Vilaflor de Chasna"],
  ["El Pinar", "El Pinar de El Hierro"],
]);

// Traduce codigos ISTAC de flujo energetico a columnas de consumo por sector.
const demandColumnMap = new Map([
  ["_T", "demand_total_mwh"],
  ["INDUSTRIA", "demand_industria_mwh"],
  ["RESIDENCIAL", "demand_residencial_mwh"],
  ["SERVICIOS", "demand_servicios_mwh"],
]);

// Define como se valida y agrega cada variable meteorologica.
// La direccion media del viento usa media circular porque 359 y 1 grados
// estan cerca, aunque numericamente parezcan extremos.
const weatherRules = {
  "Air temperature (avg.)": { column: "temp_avg_c", method: "mean", min: -20, max: 60 },
  "Air temperature (max.)": { column: "temp_max_c", method: "max", min: -20, max: 65 },
  "Air temperature (min.)": { column: "temp_min_c", method: "min", min: -30, max: 55 },
  "Atmosferic pressure (avg.)": { column: "pressure_avg_hpa", method: "mean", min: 800, max: 1100 },
  "Dew point (avg.)": { column: "dew_point_avg_c", method: "mean", min: -30, max: 40 },
  "Precipitation intensity": { column: "precip_intensity_avg_mm", method: "mean", min: 0, max: 500 },
  "Rain (daily accumulated)": { column: "rain_daily_mm", method: "max", min: 0, max: 1000 },
  "Relative humidity (avg.)": { column: "humidity_avg_pct", method: "mean", min: 0, max: 100 },
  "Wind direction (avg.)": { column: "wind_dir_avg_deg", method: "circular_mean", min: 0, max: 360 },
  "Wind direction (max.)": { column: "wind_dir_max_deg", method: "max", min: 0, max: 360 },
  "Wind direction (sdev.)": { column: "wind_dir_sdev_deg", method: "mean", min: 0, max: 180 },
  "Wind speed (avg.)": { column: "wind_speed_avg_ms", method: "mean", min: 0, max: 75 },
  "Wind speed (max.)": { column: "wind_speed_max_ms", method: "max", min: 0, max: 100 },
  "Wind speed (sdev.)": { column: "wind_speed_sdev_ms", method: "mean", min: 0, max: 30 },
};

const weatherColumns = Object.values(weatherRules).map((rule) => rule.column);
const fixedDemandHeaders = [
  "municipality",
  "date",
  "demand_total_mwh",
  "demand_industria_mwh",
  "demand_residencial_mwh",
  "demand_servicios_mwh",
];
const fixedWeatherHeaders = ["municipality", "date", ...weatherColumns, "weather_station_count"];

function removeDiacritics(text) {
  return text.normalize("NFD").replace(/\p{Diacritic}/gu, "");
}

// Normaliza municipios: limpia espacios, quita tildes para buscar alias
// y devuelve el nombre corregido cuando existe equivalencia conocida.
function normalizeMunicipalityName(name) {
  if (!name) {
    return null;
  }
  const trimmed = name.replace(/\s+/g, " ").trim();
  const asciiKey = removeDiacritics(trimmed);
  return municipalityAliases.get(asciiKey) || trimmed;
}

// Extrae municipio e isla desde la descripcion textual de una estacion.
// Si no hay municipio claro, la estacion queda sin mapear.
function parseLocationDescription(description) {
  if (!description) {
    return { municipality: null, island: null };
  }

  const trimmed = description.trim();
  const detailed = trimmed.match(/ en (?<municipality>.+?) \((?<island>[^()]+)\)\s*$/);
  if (detailed?.groups) {
    return {
      municipality: normalizeMunicipalityName(detailed.groups.municipality),
      island: detailed.groups.island.trim(),
    };
  }

  const fallback = trimmed.match(/^(?<place>.+?) \((?<island>[^()]+)\)\s*$/);
  if (fallback?.groups) {
    return { municipality: null, island: fallback.groups.island.trim() };
  }

  return { municipality: null, island: null };
}

// Parser CSV con soporte de comillas y comillas escapadas.
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

// Devuelve un mapa nombre_columna -> posicion para acceder por nombre.
function getColumnIndexMap(headerLine) {
  const fields = parseCsvLine(headerLine);
  const map = new Map();
  fields.forEach((field, index) => {
    const cleanField = index === 0 ? field.replace(/^\uFEFF/, "") : field;
    map.set(cleanField, index);
  });
  return map;
}

function parseNumber(value) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function round4(value) {
  return value == null ? null : Math.round(value * 10000) / 10000;
}

function getOrCreateDemandRow(store, municipality, date) {
  const key = `${municipality}|${date}`;
  let row = store.get(key);
  if (!row) {
    row = { municipality, date };
    store.set(key, row);
  }
  return row;
}

// Crea acumuladores para las diferentes formas de agregacion meteorologica.
function createAggregator(method) {
  if (method === "mean") {
    return { method, sum: 0, count: 0 };
  }
  if (method === "max" || method === "min") {
    return { method, value: null };
  }
  if (method === "circular_mean") {
    return { method, sinSum: 0, cosSum: 0, count: 0 };
  }
  throw new Error(`Unsupported aggregation method: ${method}`);
}

// Incorpora una observacion valida al acumulador correspondiente.
function updateAggregator(aggregator, value) {
  if (aggregator.method === "mean") {
    aggregator.sum += value;
    aggregator.count += 1;
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
    aggregator.sinSum += Math.sin(radians);
    aggregator.cosSum += Math.cos(radians);
    aggregator.count += 1;
  }
}

// Convierte cada acumulador en el valor final que se escribira en el CSV.
function finalizeAggregator(aggregator) {
  if (aggregator.method === "mean") {
    return aggregator.count === 0 ? null : round4(aggregator.sum / aggregator.count);
  }
  if (aggregator.method === "max" || aggregator.method === "min") {
    return aggregator.value == null ? null : round4(aggregator.value);
  }
  if (aggregator.method === "circular_mean") {
    if (aggregator.count === 0) {
      return null;
    }
    let angle = (Math.atan2(aggregator.sinSum, aggregator.cosSum) * 180) / Math.PI;
    if (angle < 0) {
      angle += 360;
    }
    return round4(angle);
  }
  throw new Error(`Unsupported aggregation method: ${aggregator.method}`);
}

// Guarda observaciones por municipio-fecha y conserva las estaciones usadas.
function getOrCreateWeatherRow(store, municipality, date) {
  const key = `${municipality}|${date}`;
  let row = store.get(key);
  if (!row) {
    row = {
      municipality,
      date,
      stationIds: new Set(),
      aggregators: Object.create(null),
    };
    store.set(key, row);
  }
  return row;
}

function escapeCsv(value) {
  if (value == null) {
    return "";
  }
  const text = String(value);
  if (/[",\n\r]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

async function writeCsv(filePath, headers, rows) {
  await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
  const stream = fs.createWriteStream(filePath, { encoding: "utf8" });
  stream.write(`${headers.join(",")}\n`);
  for (const row of rows) {
    const line = headers.map((header) => escapeCsv(row[header])).join(",");
    stream.write(`${line}\n`);
  }
  await new Promise((resolve, reject) => {
    stream.end(resolve);
    stream.on("error", reject);
  });
}

async function resolveInputFile(basePath, primaryRelativePath, legacyRelativePath) {
  const primaryPath = path.join(basePath, primaryRelativePath);
  try {
    await fs.promises.access(primaryPath, fs.constants.F_OK);
    return primaryPath;
  } catch {}

  const legacyPath = path.join(basePath, legacyRelativePath);
  await fs.promises.access(legacyPath, fs.constants.F_OK);
  return legacyPath;
}

async function resolveWeatherDirectory(basePath) {
  const preferredPath = path.join(basePath, config.weatherDir);
  try {
    const preferredFiles = await fs.promises.readdir(preferredPath);
    if (preferredFiles.some((file) => config.weatherPattern.test(file))) {
      return preferredPath;
    }
  } catch {}

  return basePath;
}

// Lee el catalogo de estaciones y crea el mapeo estacion -> municipio.
// Este paso permite transformar observaciones de estaciones en variables municipales.
async function processStations(basePath) {
  console.log("Cargando estaciones meteorologicas...");
  const filePath = await resolveInputFile(basePath, config.stationsFile, config.legacyStationsFile);
  const fileContents = (await fs.promises.readFile(filePath)).toString("latin1");
  const lines = fileContents.split(/\r?\n/).filter(Boolean);
  const header = getColumnIndexMap(lines[0]);
  const stationLookup = new Map();
  const outputRows = [];

  for (let i = 1; i < lines.length; i += 1) {
    const fields = parseCsvLine(lines[i]);
    const thingId = Number.parseInt(fields[header.get("thing_id")], 10);
    const locationDescription = fields[header.get("location_description")];
    const parsedLocation = parseLocationDescription(locationDescription);
    const isMapped = Boolean(parsedLocation.municipality);

    stationLookup.set(thingId, {
      thingId,
      municipality: parsedLocation.municipality,
      island: parsedLocation.island,
      isMapped,
    });

    outputRows.push({
      thing_id: thingId,
      thing_name: fields[header.get("thing_name")],
      location_id: fields[header.get("location_id")],
      location_description: locationDescription,
      municipality: parsedLocation.municipality,
      island: parsedLocation.island,
      is_mapped: isMapped,
      date_from: fields[header.get("date_from")],
      date_to: fields[header.get("date_to")],
    });
  }

  return { stationLookup, outputRows };
}

// Procesa el fichero ISTAC y genera una fila por municipio y fecha.
// Conserva solo municipios, sectores conocidos y valores validos.
async function processDemand(basePath) {
  console.log("Procesando demanda ISTAC...");
  const demandRows = new Map();
  const filePath = await resolveInputFile(basePath, config.istacFile, config.legacyIstacFile);
  const stream = fs.createReadStream(filePath, { encoding: "utf8" });
  const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

  let header = null;
  let lineCount = 0;

  for await (const line of rl) {
    if (!header) {
      header = getColumnIndexMap(line);
      continue;
    }

    lineCount += 1;
    const fields = parseCsvLine(line);
    const territoryCode = fields[header.get("TERRITORIO_CODE")];
    if (!/^\d{5}$/.test(territoryCode)) {
      continue;
    }

    const municipality = normalizeMunicipalityName(fields[header.get("TERRITORIO#es")]);
    const date = fields[header.get("TIME_PERIOD_CODE")];
    const flow = fields[header.get("FLUJO_ENERGIA_CODE")];
    const column = demandColumnMap.get(flow);
    if (!column) {
      continue;
    }

    const value = parseNumber(fields[header.get("OBS_VALUE")]);
    if (value == null || value < 0) {
      continue;
    }

    const row = getOrCreateDemandRow(demandRows, municipality, date);
    row[column] = round4(value);

    if (lineCount % 200000 === 0) {
      console.log(`  Filas ISTAC leidas: ${lineCount}`);
    }
  }

  return demandRows;
}

// Procesa observaciones meteorologicas anuales, filtra registros validos
// y agrega cada variable por municipio y fecha.
async function processWeather(basePath, stationLookup) {
  console.log("Procesando observaciones meteorologicas...");
  const weatherRows = new Map();
  const weatherBasePath = await resolveWeatherDirectory(basePath);
  const files = (await fs.promises.readdir(weatherBasePath))
    .filter((file) => config.weatherPattern.test(file))
    .sort();

  for (const file of files) {
    console.log(`  Archivo: ${file}`);
    const filePath = path.join(weatherBasePath, file);
    const stream = fs.createReadStream(filePath, { encoding: "utf8" });
    const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

    let header = null;
    let lineCount = 0;

    for await (const line of rl) {
      if (!header) {
        header = getColumnIndexMap(line);
        continue;
      }

      lineCount += 1;
      if (!line.endsWith(",valid")) {
        continue;
      }

      const fields = parseCsvLine(line);
      const thingId = Number.parseInt(fields[header.get("thing_id")], 10);
      const station = stationLookup.get(thingId);
      if (!station || !station.municipality) {
        continue;
      }

      const datastream = fields[header.get("datastream_name")];
      const rule = weatherRules[datastream];
      if (!rule) {
        continue;
      }

      const value = parseNumber(fields[header.get("result")]);
      if (value == null || value < rule.min || value > rule.max) {
        continue;
      }

      const rawDate = fields[header.get("phenomenon_time_end")] || fields[header.get("result_time")];
      if (!rawDate || rawDate.length < 10) {
        continue;
      }
      const date = rawDate.slice(0, 10);

      const row = getOrCreateWeatherRow(weatherRows, station.municipality, date);
      row.stationIds.add(thingId);

      let aggregator = row.aggregators[rule.column];
      if (!aggregator) {
        aggregator = createAggregator(rule.method);
        row.aggregators[rule.column] = aggregator;
      }
      updateAggregator(aggregator, value);

      if (lineCount % 1000000 === 0) {
        console.log(`    Filas leidas: ${lineCount}`);
      }
    }
  }

  return { weatherRows, files };
}

// Materializa las agregaciones meteorologicas y ordena las filas.
function finalizeWeatherRows(weatherRows) {
  const finalRows = [];
  for (const row of weatherRows.values()) {
    const finalRow = {
      municipality: row.municipality,
      date: row.date,
      weather_station_count: row.stationIds.size,
    };
    for (const column of weatherColumns) {
      finalRow[column] = row.aggregators[column] ? finalizeAggregator(row.aggregators[column]) : null;
    }
    finalRows.push(finalRow);
  }
  finalRows.sort((a, b) => {
    if (a.municipality !== b.municipality) {
      return a.municipality.localeCompare(b.municipality, "es");
    }
    return a.date.localeCompare(b.date);
  });
  return finalRows;
}

// Ordena la demanda ya agrupada por municipio-fecha.
function finalizeDemandRows(demandRows) {
  const finalRows = Array.from(demandRows.values());
  finalRows.sort((a, b) => {
    if (a.municipality !== b.municipality) {
      return a.municipality.localeCompare(b.municipality, "es");
    }
    return a.date.localeCompare(b.date);
  });
  return finalRows;
}

// Une demanda y meteorologia por municipio y fecha.
// Si no hay dato meteorologico, se mantienen columnas vacias para filtrar despues.
function mergeDemandAndWeather(demandRows, weatherRows) {
  const weatherByKey = new Map(weatherRows.map((row) => [`${row.municipality}|${row.date}`, row]));
  const mergedRows = [];

  for (const demandRow of demandRows) {
    const key = `${demandRow.municipality}|${demandRow.date}`;
    const weatherRow = weatherByKey.get(key);
    const mergedRow = { ...demandRow };

    for (const column of weatherColumns) {
      mergedRow[column] = weatherRow ? weatherRow[column] : null;
    }
    mergedRow.weather_station_count = weatherRow ? weatherRow.weather_station_count : null;
    mergedRows.push(mergedRow);
  }

  return mergedRows;
}

async function main() {
  const basePath = config.basePath;
  const outputPath = path.join(basePath, config.outputDir);

  const { stationLookup, outputRows: stationRows } = await processStations(basePath);
  const demandMap = await processDemand(basePath);
  const { weatherRows: weatherMap, files: weatherFiles } = await processWeather(basePath, stationLookup);

  const demandRows = finalizeDemandRows(demandMap);
  const weatherRows = finalizeWeatherRows(weatherMap);
  const mergedRows = mergeDemandAndWeather(demandRows, weatherRows);

  await writeCsv(
    path.join(outputPath, "stations_clean.csv"),
    ["thing_id", "thing_name", "location_id", "location_description", "municipality", "island", "is_mapped", "date_from", "date_to"],
    stationRows,
  );

  await writeCsv(path.join(outputPath, "istac_daily_municipal_clean.csv"), fixedDemandHeaders, demandRows);
  await writeCsv(path.join(outputPath, "weather_daily_municipal_clean.csv"), fixedWeatherHeaders, weatherRows);
  await writeCsv(
    path.join(outputPath, "demand_weather_daily_municipal.csv"),
    [...fixedDemandHeaders, ...weatherColumns, "weather_station_count"],
    mergedRows,
  );

  console.log(`Municipios con demanda: ${new Set(demandRows.map((row) => row.municipality)).size}`);
  console.log(`Filas demanda: ${demandRows.length}`);
  console.log(`Filas meteorologia: ${weatherRows.length}`);
  console.log(`Estaciones mapeadas: ${stationRows.filter((row) => row.is_mapped).length}`);
  console.log(`Estaciones no mapeadas: ${stationRows.filter((row) => !row.is_mapped).length}`);
  console.log(`Archivos meteorologicos procesados: ${weatherFiles.length}`);

  console.log("Proceso completado.");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
