"""점 지도 HTML 생성(순수 문자열 - Streamlit 없이도 쓸 수 있어 보고서에도 같이 쓴다).

구글 지도(키 필요)와 OpenStreetMap(키 불필요) 두 가지를 지원하고, 둘 다 같은 방식으로 움직인다.
  - 원(마커)이나 오른쪽 '지점 목록'의 이름을 누르면 그 위치로 바로 확대 이동하고 설명창이 열린다.
  - 지도의 빈 곳을 누르면 그 자리로 두 단계 확대된다.
  - '전체 보기'를 누르면 모든 지점이 보이는 처음 화면으로 돌아간다.
"""
from __future__ import annotations

import html
import json
from urllib.parse import quote

import pandas as pd

FOCUS_ZOOM = 15  # 지점을 눌렀을 때 확대 수준(약 1~2km 범위)

_CSS = """html,body{margin:0;height:100%;font-family:'Noto Sans KR','Malgun Gothic',system-ui,sans-serif}
.row{display:flex;gap:10px;width:100%;height:@@H@@px}
.wrap{position:relative;flex:1;min-width:0;height:100%}
#map{width:100%;height:100%;border:1px solid #e1e1e1;border-radius:4px;box-sizing:border-box}
.legend{position:absolute;left:10px;bottom:24px;z-index:1000;background:#fffffff2;border:1px solid #e1e1e1;border-radius:4px;
 padding:6px 10px;font-size:12px;color:#666;pointer-events:none}
.mk{border-radius:50%;background:#224da0;color:#fff;font-weight:700;display:flex;align-items:center;justify-content:center;
 border:2px solid #fff;box-shadow:0 1px 4px #0004;opacity:.92;font-size:13px}
.iw{font-size:13px;line-height:1.55;min-width:180px;color:#333}.iw b{font-size:14px}
.msg{padding:20px;font-size:14px;color:#666;line-height:1.6}
.side{width:210px;flex:none;display:flex;flex-direction:column;border:1px solid #e1e1e1;border-radius:4px;overflow:hidden;background:#fff}
.side h4{margin:0;padding:8px 10px;font-size:12px;font-weight:700;color:#224da0;background:#f4f6fb;border-bottom:1px solid #e1e1e1;
 display:flex;justify-content:space-between;align-items:center}
.side button{font-size:11px;border:1px solid #d5d5d5;background:#fff;border-radius:4px;padding:2px 8px;cursor:pointer;color:#333;font-family:inherit}
.side button:hover{border-color:#224da0;color:#224da0}
.hint{padding:6px 10px;font-size:11px;color:#888;border-bottom:1px solid #f0f0f0}
.list{overflow:auto;flex:1}
.item{display:flex;gap:8px;align-items:center;padding:7px 10px;font-size:12.5px;cursor:pointer;border-bottom:1px solid #f0f0f0;color:#333}
.item:hover,.item.on{background:#eaf0fa;color:#224da0}
.item b{flex:none;min-width:22px;height:22px;border-radius:11px;background:#224da0;color:#fff;font-size:11px;
 display:flex;align-items:center;justify-content:center;padding:0 3px;box-sizing:border-box}
.item span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
@media(max-width:640px){.side{display:none}}"""

_BODY = ('<div class="row"><div class="wrap"><div id="map"></div><div class="legend">@@LEGEND@@</div></div>'
         '<div class="side"><h4>지점 목록<button id="reset" type="button">전체 보기</button></h4>'
         '<div class="hint">이름을 누르면 그 위치로 이동해요</div><div class="list" id="list"></div></div></div>')

# 지점 목록 만들기·선택 표시(두 지도 공통)
_COMMON_JS = """var PTS=@@PTS@@;
function fail(t){document.getElementById('map').innerHTML='<div class="msg">'+t+'</div>';}
function buildList(focus){var box=document.getElementById('list');
  PTS.forEach(function(p,i){var d=document.createElement('div');d.className='item';d.innerHTML='<b></b><span></span>';
    d.children[0].textContent=p.n;d.children[1].textContent=p.title;d.onclick=function(){focus(i)};box.appendChild(d);});}
function mark(i){var it=document.querySelectorAll('.item');
  it.forEach(function(e,k){e.classList.toggle('on',k===i)});if(it[i])it[i].scrollIntoView({block:'nearest'});}"""

_GOOGLE_JS = """window.gm_authFailure=function(){fail('구글 지도 인증에 실패했어요.<br>GOOGLE_MAPS_API_KEY 와, Google Cloud 콘솔의 <b>Maps JavaScript API 사용 설정 · 결제 계정 · 웹사이트 제한</b>(예: http://localhost:8501/*)을 확인하세요.');};
window.initMap=function(){
  if(typeof google==='undefined'||!google.maps){fail('구글 지도를 불러오지 못했어요. 인터넷 연결을 확인하세요.');return;}
  var map=new google.maps.Map(document.getElementById('map'),{center:{lat:33.38,lng:126.55},zoom:10,
      mapTypeControl:false,streetViewControl:false,fullscreenControl:true,gestureHandling:'cooperative',disableDoubleClickZoom:true});
  var b=new google.maps.LatLngBounds(),iw=new google.maps.InfoWindow(),markers=[];
  var avgs=PTS.map(function(p){return p.avg}),mn=Math.min.apply(null,avgs),mx=Math.max.apply(null,avgs);
  function focus(i){var p=PTS[i];map.panTo({lat:p.lat,lng:p.lon});setTimeout(function(){map.setZoom(@@Z@@)},250);
    iw.setContent(p.html);iw.open({map:map,anchor:markers[i]});mark(i);}
  PTS.forEach(function(p,i){var r=13+(mx>mn?(p.avg-mn)/(mx-mn)*10:4);
    var m=new google.maps.Marker({position:{lat:p.lat,lng:p.lon},map:map,
      icon:{path:google.maps.SymbolPath.CIRCLE,scale:r,fillColor:'#224da0',fillOpacity:.92,strokeColor:'#ffffff',strokeWeight:2},
      label:{text:String(p.n),color:'#ffffff',fontWeight:'700',fontSize:'13px'}});
    m.addListener('click',function(){focus(i)});markers.push(m);b.extend(m.getPosition());});
  function overview(){iw.close();mark(-1);if(PTS.length>1)map.fitBounds(b,40);else map.setZoom(11);}
  if(PTS.length>1)map.fitBounds(b,40);
  map.addListener('click',function(e){iw.close();mark(-1);map.panTo(e.latLng);map.setZoom(Math.min(map.getZoom()+2,17));});
  document.getElementById('reset').onclick=overview;buildList(focus);
};"""

_LEAFLET_JS = """if(typeof L==='undefined'){fail('지도를 불러오지 못했어요. 인터넷 연결을 확인하세요.');}
else{
  var map=L.map('map',{scrollWheelZoom:false,doubleClickZoom:false}).setView([33.38,126.55],10);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'&copy; OpenStreetMap'}).addTo(map);
  var avgs=PTS.map(function(p){return p.avg}),mn=Math.min.apply(null,avgs),mx=Math.max.apply(null,avgs),ll=[],markers=[];
  function focus(i){var p=PTS[i];map.flyTo([p.lat,p.lon],@@Z@@,{duration:.7});markers[i].openPopup();mark(i);}
  PTS.forEach(function(p,i){var s=26+(mx>mn?(p.avg-mn)/(mx-mn)*20:8);
    var icon=L.divIcon({className:'',html:'<div class="mk" style="width:'+s+'px;height:'+s+'px">'+p.n+'</div>',iconSize:[s,s],iconAnchor:[s/2,s/2]});
    var m=L.marker([p.lat,p.lon],{icon:icon}).addTo(map).bindPopup(p.html);
    m.on('click',function(){focus(i)});markers.push(m);ll.push([p.lat,p.lon]);});
  function overview(){map.closePopup();mark(-1);if(ll.length>1)map.flyToBounds(ll,{padding:[40,40],duration:.7});else map.flyTo([33.38,126.55],10);}
  if(ll.length>1)map.fitBounds(ll,{padding:[40,40]});
  map.on('click',function(e){mark(-1);map.flyTo(e.latlng,Math.min(map.getZoom()+2,17),{duration:.6});});
  document.getElementById('reset').onclick=overview;buildList(focus);
}"""


def _points(df: pd.DataFrame) -> list[dict]:
    """점 지도용 데이터. df 컬럼: lat, lon, 장소, 값(원 안 숫자), 크기(원 크기), 설명(줄바꿈은 \\n)."""
    pts = []
    for r in df.itertuples():
        lines = "<br>".join(html.escape(x) for x in str(r.설명).split("\n"))
        info = f"<div class='iw'><b>{html.escape(str(r.장소))}</b><br>{lines}</div>"
        pts.append(dict(lat=float(r.lat), lon=float(r.lon), n=int(r.값), avg=float(r.크기), title=str(r.장소), html=info))
    return pts


def _js(pts: list[dict]) -> str:
    return json.dumps(pts, ensure_ascii=False).replace("</", "<\\/")


def _fill(text: str, h: int, legend: str, pts: list[dict], key: str = "") -> str:
    for k, v in (("@@H@@", str(int(h))), ("@@LEGEND@@", html.escape(legend)), ("@@PTS@@", _js(pts)),
                 ("@@Z@@", str(FOCUS_ZOOM)), ("@@KEY@@", quote(key, safe=""))):
        text = text.replace(k, v)
    return text


def google_map_html(df: pd.DataFrame, key: str, height: int = 480, legend: str = "") -> str:
    pts = _points(df)
    doc = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<style>{_CSS}</style></head><body>{_BODY}<script>{_COMMON_JS}\n{_GOOGLE_JS}</script>'
           '<script async src="https://maps.googleapis.com/maps/api/js?key=@@KEY@@&callback=initMap&language=ko&region=KR" '
           "onerror=\"fail('구글 지도 스크립트를 불러오지 못했어요. 인터넷 연결을 확인하세요.')\"></script></body></html>")
    return _fill(doc, height, legend, pts, key)


def leaflet_map_html(df: pd.DataFrame, legend: str = "", height: int = 480) -> str:
    """API 키가 필요 없는 지도(OpenStreetMap). 인터넷 연결이 있으면 표시된다."""
    pts = _points(df)
    doc = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           '<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">'
           f'<style>{_CSS}</style></head><body>{_BODY}'
           '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>'
           f'<script>{_COMMON_JS}\n{_LEAFLET_JS}</script></body></html>')
    return _fill(doc, height, legend, pts)
