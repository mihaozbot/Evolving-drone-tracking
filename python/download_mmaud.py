# mmaud ground truth (leica, 5 Hz): bag -> csv
import http.cookiejar
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://entuedu-my.sharepoint.com/:u:/g/personal/shyuan_staff_main_ntu_edu_sg/"
DRONES = {
    "mavic2": "EW4eZm1eEm9FndMN8hxQyesBNdQny6iV_n86Jtwcysi7Lw?e=s8csWh",
    "mavic3": "Eaol9EuQTZ5BmYffvBSo3kIB0DwKDcYstGvwg-Qrtnts4A?e=ZaJpsa",
    "phantom4": "EYlHSi2RmehBoPYDXiKspL0BsYx0IDXFtmG_UcFRl_AKBQ?e=B7KLP3",
    "avata": "EYoZq-lAmvVMgncDhDX4MDIBy6_mqqobYE1WW8xQ7EhfxQ?e=QJqwJc",
    "m300": "ERDR8BNyj0hBrmsCIg8rP8wBnmz52qq4WTqjkCc2dnUQAw?e=KkXYhl",
}
DATA = Path(__file__).resolve().parent.parent / "data" / "mmaud"


def load(drone, dt=0.2):
    d = np.loadtxt(DATA / f"{drone}.csv", delimiter=",", skiprows=1)
    tt = np.arange(0.0, d[-1, 0], dt)
    return np.column_stack([np.interp(tt, d[:, 0], d[:, i]) for i in (1, 2, 3)])


def main():
    from rosbags.highlevel import AnyReader
    DATA.mkdir(parents=True, exist_ok=True)
    for drone, link in DRONES.items():
        bag = DATA / f"{drone}.bag"
        if not bag.exists():
            # sharepoint wants the cookie first
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
            opener.addheaders = [("User-Agent", "Mozilla/5.0")]
            opener.open(BASE + link).read()
            bag.write_bytes(opener.open(BASE + link + "&download=1").read())
        rows = []
        with AnyReader([bag]) as reader:
            con = [c for c in reader.connections if c.topic == "/leica/point/relative"]
            for c, _, raw in reader.messages(connections=con):
                m = reader.deserialize(raw, c.msgtype)
                rows.append((m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, m.point.x, m.point.y, m.point.z))
        rows = np.array(sorted(rows))
        rows[:, 0] -= rows[0, 0]
        np.savetxt(DATA / f"{drone}.csv", rows, delimiter=",", header="t,x,y,z", comments="", fmt="%.4f")
        print(f"{drone:9s} {rows[-1, 0]:6.1f} s")


if __name__ == "__main__":
    main()
