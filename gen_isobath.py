#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_isobath.py  ―  GMRT / GEBCO の海底地形グリッドから等深線(isobath)を生成する。

入力 : GMRT または GEBCO からダウンロードした 1 枚のグリッド
        - GeoTIFF (.tif / .tiff)   … rasterio が必要 (pip install rasterio)
        - ESRI ASCII (.asc)        … 追加ライブラリ不要 (numpy のみ)
出力 : sagami_isobath.geojson      … 等深線 (LineString, property: depth=負の水深[m])
        isobath_data.js            … HTML から読み込む用 `const ISOBATH = {...geojson...}`

使い方:
    python3 gen_isobath.py sagami_bathy.tif
    python3 gen_isobath.py sagami_bathy.asc --levels -10 -20 -30 -50 -100 -200 -500 -1000

生成した isobath_data.js を HTML に <script src="isobath_data.js"></script> で読み込み、
HANDOFF.md の「等深線オーバーレイ追加スニペット」を main script に貼れば地図に出ます。

ダウンロード元(相模湾の範囲: lon 139.30–139.85 / lat 35.00–35.36):
  GMRT GeoTIFF:
   https://www.gmrt.org/services/GridServer?minlongitude=139.30&maxlongitude=139.85&minlatitude=35.00&maxlatitude=35.36&format=geotiff&resolution=max
  GMRT ESRI ASCII:
   https://www.gmrt.org/services/GridServer?minlongitude=139.30&maxlongitude=139.85&minlatitude=35.00&maxlatitude=35.36&format=esriascii&resolution=high
  GEBCO (代替): https://download.gebco.net/
"""
import sys, json, argparse
import numpy as np

DEFAULT_LEVELS = [-10, -20, -30, -50, -100, -200, -300, -500, -1000, -1500]


def load_grid(path):
    """戻り値: lon1d (W,昇順), lat1d (H,昇順), Z (H,W) 標高[m] 北が上→latは昇順に整列して返す"""
    pl = path.lower()
    if pl.endswith(('.tif', '.tiff')):
        import rasterio
        with rasterio.open(path) as ds:
            Z = ds.read(1).astype('float64')
            nodata = ds.nodata
            if nodata is not None:
                Z[Z == nodata] = np.nan
            H, W = Z.shape
            tr = ds.transform
            # ピクセル中心の経度緯度
            cols = np.arange(W); rows = np.arange(H)
            lon1d = tr.c + (cols + 0.5) * tr.a          # tr.a = +cellsize
            lat1d = tr.f + (rows + 0.5) * tr.e          # tr.e = -cellsize (北が上)
        # lat を昇順に
        if lat1d[0] > lat1d[-1]:
            lat1d = lat1d[::-1]; Z = Z[::-1, :]
        return lon1d, lat1d, Z

    # ---- ESRI ASCII ----
    with open(path) as f:
        hdr = {}
        for _ in range(6):
            k, v = f.readline().split()
            hdr[k.lower()] = float(v)
        ncols = int(hdr['ncols']); nrows = int(hdr['nrows'])
        xll = hdr.get('xllcorner', hdr.get('xllcenter'))
        yll = hdr.get('yllcorner', hdr.get('yllcenter'))
        cs = hdr['cellsize']; nod = hdr.get('nodata_value', -9999.0)
        Z = np.loadtxt(f).reshape(nrows, ncols).astype('float64')
    Z[Z == nod] = np.nan
    half = 0.0 if ('xllcenter' in hdr) else cs / 2.0
    lon1d = xll + half + np.arange(ncols) * cs
    lat1d = yll + half + np.arange(nrows) * cs          # yll は南端
    Z = Z[::-1, :]                                       # ASCII は北が先頭→南が先頭へ
    return lon1d, lat1d, Z


def contours_to_features(lon1d, lat1d, Z, levels, simplify_deg=0.0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    LON, LAT = np.meshgrid(lon1d, lat1d)
    Zf = np.where(np.isnan(Z), 9e9, Z)                   # NaN を陸側扱いで充填
    fig = plt.figure()
    cs = plt.contour(LON, LAT, Zf, levels=sorted(levels))
    feats = []
    for lev, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            if len(seg) < 2:
                continue
            pts = seg
            if simplify_deg > 0:
                pts = _rdp(np.asarray(seg), simplify_deg)
            coords = [[round(float(x), 6), round(float(y), 6)] for x, y in pts]
            feats.append({
                "type": "Feature",
                "properties": {"depth": int(lev)},
                "geometry": {"type": "LineString", "coordinates": coords},
            })
    plt.close(fig)
    return feats


def _rdp(pts, eps):
    if len(pts) < 3:
        return pts
    s, e = pts[0], pts[-1]
    d = e - s; L = np.hypot(*d)
    if L == 0:
        dist = np.hypot(*(pts - s).T)
    else:
        dist = np.abs(np.cross(np.tile(d, (len(pts), 1)), pts - s)) / L
    i = int(np.argmax(dist))
    if dist[i] > eps:
        return np.vstack([_rdp(pts[:i + 1], eps)[:-1], _rdp(pts[i:], eps)])
    return np.vstack([s, e])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('grid')
    ap.add_argument('--levels', type=float, nargs='+', default=DEFAULT_LEVELS)
    ap.add_argument('--simplify', type=float, default=0.0002,
                    help='RDP 簡略化の許容誤差(度)。0で無効。既定 ~20m')
    ap.add_argument('--out', default='sagami_isobath.geojson')
    a = ap.parse_args()

    lon1d, lat1d, Z = load_grid(a.grid)
    print(f"grid: {Z.shape[1]} x {Z.shape[0]}  lon[{lon1d.min():.4f},{lon1d.max():.4f}] "
          f"lat[{lat1d.min():.4f},{lat1d.max():.4f}]  z[{np.nanmin(Z):.0f},{np.nanmax(Z):.0f}]m")
    feats = contours_to_features(lon1d, lat1d, Z, a.levels, a.simplify)
    gj = {"type": "FeatureCollection", "name": "Sagami Bay Isobaths",
          "features": feats}
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(gj, f, ensure_ascii=False)
    with open('isobath_data.js', 'w', encoding='utf-8') as f:
        f.write('const ISOBATH=' + json.dumps(gj, ensure_ascii=False) + ';')
    bylev = {}
    for ft in feats:
        bylev[ft['properties']['depth']] = bylev.get(ft['properties']['depth'], 0) + 1
    print('lines per level:', dict(sorted(bylev.items(), reverse=True)))
    print('wrote', a.out, 'and isobath_data.js  (', len(feats), 'lines )')


if __name__ == '__main__':
    main()
