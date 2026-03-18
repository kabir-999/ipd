import { useEffect, useMemo, useState } from 'react';
import L from 'leaflet';
import 'leaflet-draw';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';
import { MapContainer, Rectangle, TileLayer, useMap, useMapEvents } from 'react-leaflet';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8001';
const MAX_YEAR_RANGE = 10;
const DEFAULT_CENTER = [20.5937, 78.9629];
const DEFAULT_ZOOM = 5;
const MONTHS = [
  { value: 1, label: 'January' },
  { value: 2, label: 'February' },
  { value: 3, label: 'March' },
  { value: 4, label: 'April' },
  { value: 5, label: 'May' },
  { value: 6, label: 'June' },
  { value: 7, label: 'July' },
  { value: 8, label: 'August' },
  { value: 9, label: 'September' },
  { value: 10, label: 'October' },
  { value: 11, label: 'November' },
  { value: 12, label: 'December' }
];
const DAYS = Array.from({ length: 31 }, (_, index) => index + 1);

async function fetchJson(url, options = {}, timeoutMs = 600000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || `Request failed (${response.status})`);
    }

    return data;
  } catch (error) {
    if (error?.name === 'AbortError') {
      throw new Error('Analysis request timed out. Please reduce AOI/year range and try again.');
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

function bboxToLeafletBounds(bbox) {
  if (!bbox || bbox.length !== 4) return null;
  const [minLon, minLat, maxLon, maxLat] = bbox;
  return [
    [minLat, minLon],
    [maxLat, maxLon]
  ];
}

function makePointBbox(lat, lon) {
  const delta = 0.05;
  return [lon - delta, lat - delta, lon + delta, lat + delta];
}

function toChartData(result) {
  if (!result) return [];
  const rows = getFeatureRows(result);
  return rows.map((row) => ({
    year: row.year,
    ndvi: row.ndvi,
    ndwi: row.ndwi,
    vegetation: row.vegetationPercent
  }));
}

function getFeatureRows(result) {
  if (!result) return [];

  if (Array.isArray(result.features)) {
    return result.features;
  }

  if (Array.isArray(result.years)) {
    return result.years.map((year, index) => ({
      year,
      ndvi: Array.isArray(result.ndvi) ? Number(result.ndvi[index] ?? 0) : 0,
      ndwi: 0,
      savi: 0,
      evi: 0,
      ndmi: 0,
      nbr: 0,
      vegetationPercent: Array.isArray(result.vegetation) ? Number(result.vegetation[index] ?? 0) : 0
    }));
  }

  return [];
}

function getYearVisualRows(result, featureRows) {
  if (!result || !result.maps) return [];
  return featureRows
    .map((row) => {
      const yearKey = String(row.year);
      const mapSet = result.maps?.[yearKey];
      if (!mapSet) return null;
      return {
        year: row.year,
        ndviTileUrl: mapSet.ndviTileUrl || '',
        ndviUrl: mapSet.ndvi || '',
        vegetationPercent: row.vegetationPercent ?? 0
      };
    })
    .filter(Boolean);
}

function riskLabel(risk) {
  if (risk >= 0.7) return 'High Risk';
  if (risk >= 0.4) return 'Moderate Risk';
  return 'Low Risk';
}

async function geocodeDestination(query) {
  const url = `https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(query)}`;
  const response = await fetch(url, { headers: { Accept: 'application/json' } });
  if (!response.ok) {
    throw new Error('Destination search failed.');
  }
  const data = await response.json();
  if (!data.length) {
    throw new Error('Destination not found.');
  }
  return [Number(data[0].lat), Number(data[0].lon)];
}

function AOISelector({ onSelect }) {
  useMapEvents({
    click(event) {
      const { lat, lng } = event.latlng;
      onSelect(makePointBbox(lat, lng));
    }
  });

  return null;
}

function MapNavigator({ center }) {
  const map = useMap();

  useEffect(() => {
    if (center) {
      map.setView(center, 11);
    }
  }, [map, center]);

  return null;
}

function MapFitter({ bounds }) {
  const map = useMap();

  useEffect(() => {
    if (bounds) {
      map.fitBounds(bounds, { padding: [8, 8] });
    }
  }, [map, bounds]);

  return null;
}

function YearNdviMap({ tileUrl, bounds }) {
  const fallbackCenter = bounds
    ? [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
    : DEFAULT_CENTER;

  return (
    <MapContainer
      className="year-map-view"
      center={fallbackCenter}
      zoom={10}
      scrollWheelZoom={false}
      attributionControl={false}
      zoomControl={false}
    >
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" opacity={0.5} />
      <TileLayer url={tileUrl} opacity={0.85} />
      {bounds && <Rectangle bounds={bounds} pathOptions={{ color: '#d32f2f', weight: 1 }} />}
      <MapFitter bounds={bounds} />
    </MapContainer>
  );
}

function DrawControl({ onSelect, onClear }) {
  const map = useMap();

  useEffect(() => {
    const drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);

    const drawControl = new L.Control.Draw({
      position: 'topright',
      edit: {
        featureGroup: drawnItems,
        edit: false,
        remove: true
      },
      draw: {
        rectangle: true,
        polygon: false,
        polyline: false,
        circle: false,
        marker: false,
        circlemarker: false
      }
    });

    map.addControl(drawControl);

    const onCreated = (event) => {
      if (event.layerType !== 'rectangle') return;
      drawnItems.clearLayers();
      drawnItems.addLayer(event.layer);

      const bounds = event.layer.getBounds();
      const southWest = bounds.getSouthWest();
      const northEast = bounds.getNorthEast();
      onSelect([southWest.lng, southWest.lat, northEast.lng, northEast.lat]);
    };

    const onDeleted = () => {
      onClear();
    };

    map.on(L.Draw.Event.CREATED, onCreated);
    map.on(L.Draw.Event.DELETED, onDeleted);

    return () => {
      map.off(L.Draw.Event.CREATED, onCreated);
      map.off(L.Draw.Event.DELETED, onDeleted);
      map.removeControl(drawControl);
      map.removeLayer(drawnItems);
    };
  }, [map, onSelect, onClear]);

  return null;
}

export default function App() {
  const currentYear = new Date().getFullYear();
  const [destination, setDestination] = useState('');
  const [mapCenter, setMapCenter] = useState(DEFAULT_CENTER);
  const [startYear, setStartYear] = useState(currentYear - 3);
  const [endYear, setEndYear] = useState(currentYear);
  const [month, setMonth] = useState(new Date().getMonth() + 1);
  const [day, setDay] = useState(new Date().getDate());
  const [bbox, setBbox] = useState(null);
  const [resultBbox, setResultBbox] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const chartData = useMemo(() => toChartData(result), [result]);
  const featureRows = useMemo(() => getFeatureRows(result), [result]);
  const yearVisualRows = useMemo(() => getYearVisualRows(result, featureRows), [result, featureRows]);
  const selectionBounds = useMemo(() => bboxToLeafletBounds(bbox), [bbox]);
  const resultBounds = useMemo(() => bboxToLeafletBounds(resultBbox), [resultBbox]);

  function validateInputs() {
    const start = Number(startYear);
    const end = Number(endYear);

    if (!Number.isInteger(start) || !Number.isInteger(end)) {
      setError('Start year and end year must be integers.');
      return false;
    }

    const selectedMonth = Number(month);
    if (!Number.isInteger(selectedMonth) || selectedMonth < 1 || selectedMonth > 12) {
      setError('Month must be between 1 and 12.');
      return false;
    }

    const selectedDay = Number(day);
    if (!Number.isInteger(selectedDay) || selectedDay < 1 || selectedDay > 31) {
      setError('Day must be between 1 and 31.');
      return false;
    }

    if (start < 2015 || end > currentYear) {
      setError(`Year must be between 2015 and ${currentYear}.`);
      return false;
    }

    if (start >= end) {
      setError('End year must be greater than start year.');
      return false;
    }

    if (end - start > MAX_YEAR_RANGE) {
      setError(`Please select at most ${MAX_YEAR_RANGE} years.`);
      return false;
    }

    if (!bbox) {
      setError('Select an AOI by clicking a point or drawing a rectangle on the map.');
      return false;
    }

    return true;
  }

  async function goToDestination() {
    if (!destination.trim()) {
      setError('Enter a destination to search.');
      return;
    }
    try {
      setError('');
      const center = await geocodeDestination(destination.trim());
      setMapCenter(center);
    } catch (err) {
      setError(err.message || 'Destination search failed');
    }
  }

  async function runAnalysis() {
    setError('');
    setResult(null);
    setResultBbox(null);

    if (!validateInputs()) return;

    const analysisBbox = [...bbox];
    setLoading(true);
    try {
      const data = await fetchJson(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bbox: analysisBbox,
          startYear: Number(startYear),
          endYear: Number(endYear),
          month: Number(month),
          day: Number(day)
        })
      });
      setResult(data);
      setResultBbox(analysisBbox);
    } catch (err) {
      setError(err.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="layout-root">
      <div id="controls">
        <b>Interface</b>
        <label>
          Destination
          <input
            type="text"
            value={destination}
            onChange={(event) => setDestination(event.target.value)}
            placeholder="Search city or area"
          />
        </label>
        <label>
          Year 1
          <input
            type="number"
            min="2015"
            max={currentYear}
            value={startYear}
            onChange={(event) => setStartYear(event.target.value)}
          />
        </label>
        <label>
          Year 2
          <input
            type="number"
            min="2015"
            max={currentYear}
            value={endYear}
            onChange={(event) => setEndYear(event.target.value)}
          />
        </label>
        <label>
          Month
          <select value={month} onChange={(event) => setMonth(event.target.value)}>
            {MONTHS.map((entry) => (
              <option key={entry.value} value={entry.value}>
                {entry.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Day
          <select value={day} onChange={(event) => setDay(event.target.value)}>
            {DAYS.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <button type="button" onClick={goToDestination} disabled={loading}>Go to Place</button>
        <button onClick={runAnalysis} disabled={loading}>{loading ? 'Running...' : 'Run Analysis'}</button>
      </div>

      <div id="main-container">
        <div id="map-wrap">
          <MapContainer id="map" center={DEFAULT_CENTER} zoom={DEFAULT_ZOOM} scrollWheelZoom>
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            <MapNavigator center={mapCenter} />
            <AOISelector onSelect={setBbox} />
            <DrawControl onSelect={setBbox} onClear={() => setBbox(null)} />

            {selectionBounds && <Rectangle bounds={selectionBounds} pathOptions={{ color: '#d32f2f', weight: 2 }} />}
          </MapContainer>
        </div>

        <div id="output-panel">
          <h3>Model Output</h3>
          <p className="hint">Click any point or draw a rectangle to select AOI.</p>
          {error && <p className="error">{error}</p>}

          {result && (
            <>
              <div className="risk-card">
                <p className="risk-value">{(result.risk * 100).toFixed(1)}%</p>
                <p className="risk-label">{riskLabel(result.risk)}</p>
              </div>

              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="year" />
                    <YAxis yAxisId="left" domain={[0, 100]} />
                    <YAxis yAxisId="right" orientation="right" domain={[-1, 1]} />
                    <Tooltip />
                    <Legend />
                    <Line yAxisId="left" type="monotone" dataKey="vegetation" name="Vegetation %" stroke="#2e7d32" strokeWidth={2} />
                    <Line yAxisId="right" type="monotone" dataKey="ndvi" name="NDVI" stroke="#1565c0" strokeWidth={2} />
                    <Line yAxisId="right" type="monotone" dataKey="ndwi" name="NDWI" stroke="#6a1b9a" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Year</th>
                      <th>NDVI</th>
                      <th>Veg%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {featureRows.map((row) => (
                      <tr key={row.year}>
                        <td>{row.year}</td>
                        <td>{row.ndvi}</td>
                        <td>{row.vegetationPercent}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {yearVisualRows.length > 0 && (
                <div className="year-map-grid">
                  {yearVisualRows.map((row) => (
                    <div className="year-map-card" key={`maps-${row.year}`}>
                      <h3>NDVI Map - {row.year}</h3>
                      {row.ndviTileUrl ? (
                        <YearNdviMap tileUrl={row.ndviTileUrl} bounds={resultBounds} />
                      ) : (
                        <img src={`${API_BASE}${row.ndviUrl}`} alt={`NDVI map ${row.year}`} loading="lazy" />
                      )}
                      <p className="coverage-text">Vegetation Cover: {Number(row.vegetationPercent).toFixed(2)}%</p>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
