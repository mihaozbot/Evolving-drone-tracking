function download_mmaud(folder)
% mmaud ground truth (leica, 5 hz): bag -> csv, needs ros toolbox

base = "https://entuedu-my.sharepoint.com/:u:/g/personal/shyuan_staff_main_ntu_edu_sg/";
drones = ["mavic2",   "EW4eZm1eEm9FndMN8hxQyesBNdQny6iV_n86Jtwcysi7Lw?e=s8csWh";
          "mavic3",   "Eaol9EuQTZ5BmYffvBSo3kIB0DwKDcYstGvwg-Qrtnts4A?e=ZaJpsa";
          "phantom4", "EYlHSi2RmehBoPYDXiKspL0BsYx0IDXFtmG_UcFRl_AKBQ?e=B7KLP3";
          "avata",    "EYoZq-lAmvVMgncDhDX4MDIBy6_mqqobYE1WW8xQ7EhfxQ?e=QJqwJc";
          "m300",     "ERDR8BNyj0hBrmsCIg8rP8wBnmz52qq4WTqjkCc2dnUQAw?e=KkXYhl"];
if nargin < 1, folder = data_folder(); end
if ~isfolder(folder), mkdir(folder); end
for i = 1:size(drones, 1)
    drone = drones(i, 1); link = base + drones(i, 2);
    csv = fullfile(folder, drone + ".csv");
    if isfile(csv), fprintf("%-9s already there\n", drone); continue, end
    bag = fullfile(folder, drone + ".bag");
    if ~isfile(bag)
        % sharepoint wants the cookie first
        websave(bag, link + "&download=1", weboptions("HeaderFields", ...
            ["Cookie", share_cookie(link); "User-Agent", "Mozilla/5.0"], "Timeout", 300));
    end
    msgs = readMessages(select(rosbagreader(bag), "Topic", "/leica/point/relative"), "DataFormat", "struct");
    rows = cellfun(@(m) [double(m.Header.Stamp.Sec) + 1e-9 * double(m.Header.Stamp.Nsec), ...
                         m.Point.X, m.Point.Y, m.Point.Z], msgs, "UniformOutput", false);
    rows = sortrows(vertcat(rows{:}));
    rows(:, 1) = rows(:, 1) - rows(1, 1);
    writetable(array2table(round(rows, 4), "VariableNames", ["t", "x", "y", "z"]), csv);
    fprintf("%-9s %6.1f s\n", drone, rows(end, 1));
end
end


function cookie = share_cookie(link)
import matlab.net.http.*
request = RequestMessage(RequestMethod.GET, HeaderField("User-Agent", "Mozilla/5.0"));
[~, ~, history] = request.send(matlab.net.URI(link), HTTPOptions("MaxRedirects", 20));
parts = strings(0);
for h = history
    if isempty(h.Response), continue, end
    for f = h.Response.getFields("Set-Cookie")
        parts(end + 1) = extractBefore(string(f.Value) + ";", ";");
    end
end
cookie = strjoin(parts, "; ");
end
