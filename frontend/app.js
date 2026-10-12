const $ = id => document.getElementById(id);
let map, incidentLayer, selectedMarker;
let selected = { lat: 11.0168, lon: 76.9558 };
function fmt(v) { return Number.isFinite(Number(v)) ? Number(v).toLocaleString() : "—"; }
function esc(v) { return String(v == null ? "" : v).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
function initMap() {
  map = L.map("map").setView([selected.lat, selected.lon], 11);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom:19, attribution:'&copy; OpenStreetMap contributors' }).addTo(map);
  incidentLayer = L.layerGroup().addTo(map);
  selectedMarker = L.marker([selected.lat, selected.lon], {draggable:true}).addTo(map).bindPopup("Selected point — no risk score is inferred from this coordinate.");
  selectedMarker.on("dragend", e => setSelected(e.target.getLatLng().lat, e.target.getLatLng().lng));
  map.on("click", e => setSelected(e.latlng.lat, e.latlng.lng));
  updateCoords();
}
function setSelected(lat, lon) {
  selected = {lat:lat, lon:lon}; selectedMarker.setLatLng([lat,lon]); updateCoords();
  $("locationResult").textContent = "Point selected. Historical records can be checked only when the dataset contains valid coordinates. This prototype does not infer an accident probability from a map click.";
}
function updateCoords() { $("selectedLat").textContent=selected.lat.toFixed(5); $("selectedLon").textContent=selected.lon.toFixed(5); }
async function api(url, options) {
  options = options || {};
  options.headers = Object.assign({"Content-Type":"application/json"}, options.headers || {});
  const res = await fetch(url, options); const body = await res.json().catch(()=>({}));
  if (!res.ok) throw new Error(body.detail || "Request failed ("+res.status+")");
  return body;
}
async function loadSummary() {
  const d = await api("/api/summary");
  $("totalRecords").textContent=fmt(d.total_records); $("mappedRecords").textContent=fmt(d.mapped_records);
  $("modelAccuracy").textContent=d.model_metrics ? (d.model_metrics.accuracy*100).toFixed(1)+"%" : "—";
  $("modelF1").textContent=d.model_metrics ? d.model_metrics.macro_f1.toFixed(3) : "—";
  $("datasetNotice").classList.toggle("ready", !!d.dataset_loaded);
  $("datasetNotice").innerHTML=d.dataset_loaded
    ? "<b>Dataset loaded</b><span>"+fmt(d.total_records)+" rows found. "+fmt(d.mapped_records)+" rows have valid coordinates. Verify source and geography before interpreting results.</span>"
    : "<b>Dataset not loaded</b><span>"+esc(d.message)+"</span>";
  $("targetColumn").innerHTML=d.columns && d.columns.length ? '<option value="">Choose target column…</option>'+d.columns.map(c=>'<option value="'+esc(c)+'">'+esc(c)+'</option>').join("") : '<option value="">Load a dataset first</option>';
  await loadIncidents(); await inspectDataset();
}
async function loadIncidents() {
  const d=await api("/api/incidents?limit=3000"); incidentLayer.clearLayers();
  if(d.mapped && d.incidents.length) {
    const bounds=[];
    d.incidents.forEach(item=>{
      const title=Object.entries(item.properties||{}).slice(0,5).map(pair=>"<b>"+esc(pair[0])+":</b> "+esc(pair[1] == null ? "Unknown" : pair[1])).join("<br>");
      L.circleMarker([item.lat,item.lon],{radius:5,color:"#ff8e83",weight:1,fillColor:"#ff6f61",fillOpacity:.75}).bindPopup(title||"Recorded incident").addTo(incidentLayer);
      bounds.push([item.lat,item.lon]);
    });
    $("mapStatus").textContent=fmt(d.incidents.length)+" incident pins shown";
    if(bounds.length) map.fitBounds(bounds,{padding:[25,25],maxZoom:13});
  } else $("mapStatus").textContent=d.message || "No mappable records available";
}
async function inspectDataset() {
  const d=await api("/api/dataset/inspect");
  $("columnsList").innerHTML=!d.loaded ? '<div class="empty">'+esc(d.message)+'</div>' : d.columns.map(c=>'<div class="column-chip">'+esc(c.name)+'<small>'+esc(c.dtype)+' · '+fmt(c.unique)+' unique</small></div>').join("");
}
$("refreshBtn").addEventListener("click", async()=>{ $("refreshBtn").disabled=true; try{await loadSummary();}catch(e){showError(e);}finally{$("refreshBtn").disabled=false;}});
$("inspectBtn").addEventListener("click", async()=>{
  const result=$("locationResult"), label=$("locationName").value.trim(); result.textContent="Checking loaded incident coordinates…";
  try {
    const d=await api("/api/incidents?limit=10000");
    if(!d.mapped){result.textContent="Cannot inspect historical incidents by point: this dataset has no recognized latitude/longitude columns. Add a geocoded dataset before making point-specific claims.";return;}
    const near=d.incidents.map(i=>Object.assign({},i,{distanceKm:distance(selected.lat,selected.lon,i.lat,i.lon)})).filter(i=>i.distanceKm<=2).sort((a,b)=>a.distanceKm-b.distanceKm);
    if(!near.length){result.textContent="No loaded incident points were found within 2 km"+(label ? " of "+label : "")+". This is not proof that the road is safe; dataset coverage may be incomplete.";return;}
    result.innerHTML="<b>"+near.length+" recorded point(s) within 2 km"+(label ? " of "+esc(label) : "")+"</b><br>"+near.slice(0,5).map((i,n)=>(n+1)+". "+i.distanceKm.toFixed(2)+" km away — "+esc(Object.entries(i.properties||{}).slice(0,2).map(p=>p[0]+": "+(p[1] == null ? "Unknown" : p[1])).join(", ")||"Incident record")).join("<br>")+"<br><br><span style='color:#d8bd85'>Historical proximity only, not an accident probability or live safety assessment.</span>";
  } catch(e){result.textContent=e.message;}
});
function distance(a,b,c,d){const r=x=>x*Math.PI/180,R=6371,da=r(c-a),db=r(d-b),q=Math.sin(da/2)**2+Math.cos(r(a))*Math.cos(r(c))*Math.sin(db/2)**2;return R*2*Math.atan2(Math.sqrt(q),Math.sqrt(1-q));}
$("trainBtn").addEventListener("click",async()=>{
  const target=$("targetColumn").value,out=$("trainResult");
  if(!target){out.textContent="Choose a target column first.";return;}
  $("trainBtn").disabled=true;out.textContent="Training Random Forest and evaluating on a holdout split…";
  try {
    const d=await api("/api/model/train",{method:"POST",body:JSON.stringify({target_column:target})}),m=d.metrics;
    out.innerHTML="<b>Model trained</b><br>Target: "+esc(m.target)+"<br>Holdout accuracy: "+(m.accuracy*100).toFixed(1)+"%<br>Macro F1: "+m.macro_f1.toFixed(3)+"<br>Training rows: "+fmt(m.train_rows)+" · Test rows: "+fmt(m.test_rows)+"<br>Classes: "+m.classes.map(esc).join(", ")+"<br><br><span style='color:#d8bd85'>"+esc(m.warning)+"</span>";
    await loadSummary();
  }catch(e){out.textContent="Training error: "+e.message;}finally{$("trainBtn").disabled=false;}
});
function showError(e){$("datasetNotice").classList.remove("ready");$("datasetNotice").innerHTML="<b>Could not load data</b><span>"+esc(e.message)+". Check that the backend is running.</span>";}
initMap(); loadSummary().catch(showError);
