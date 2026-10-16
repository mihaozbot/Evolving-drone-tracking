function behaviour()
% demo: 2 drones at a time, sensor loses both once, then who is who

%% ---- setup ----
DT = 0.2; H = 20; WINDOW = 50; GAP = 20;
c = constants();
names = ["mavic2", "mavic3", "phantom4", "avata", "m300"];
ne = numel(names);
flights = arrayfun(@(d) load_mmaud(d, DT), names, "UniformOutput", false);
centre = mean(vertcat(flights{:}), 1);
for i = 1:ne
    flights{i} = (flights{i} - centre) * turn(i) + centre;
end
allp = vertcat(flights{:}); lo = min(allp); hi = max(allp);
rng(0);
z = cellfun(@(f) f + 0.05 * randn(size(f)), flights, "UniformOutput", false);  % fake sensor noise
track = cell(1, ne);
for e = 1:ne
    track{e} = EvoKF(DT, "q", 0.1, "budget", 6);
    track{e}.start(z{e}(1, :));
end
clf = Swarms("budget", 12, "dim", numel(c.FEATURES), "classes", ne);
pairs = nchoosek(1:ne, 2); pairs = pairs(randperm(size(pairs, 1)), :);
at = 2 * ones(1, ne);
steps = round(30 / DT);
later = struct("t", {}, "key", {}, "f", {}, "e", {});
logm = zeros(0, 3); ids = zeros(0, 5); frames = struct("clock", {}, "state", {});
clock = 0; windows = zeros(0, 3); flown = repmat({zeros(0, 2)}, 1, ne);
record.vedenje = [1 2]; record.gibanje = [1 2];

%% ---- main loop ----
ip = 0;
while true
    ip = mod(ip, size(pairs, 1)) + 1;
    left = arrayfun(@(e) at(e) + steps + H <= size(flights{e}, 1), 1:ne);
    if sum(left) < 2, break, end
    pair = pairs(ip, :); pair = pair(left(pair));
    if numel(pair) < 2, continue, end
    starts = 30:5:80;
    speed = arrayfun(@(j) sum(arrayfun(@(e) norm(flights{e}(at(e) + j + 1, 1:2) - flights{e}(at(e) + j, 1:2)), pair)), starts);
    [~, i] = max(speed); L0 = starts(i); L1 = L0 + GAP;  % lose them where they fly fastest
    who = pair; seen = [0 0]; Lp = zeros(2); known = {};
    for j = 0:steps - 1
        clock = clock + 1;
        lost = j >= L0 && j < L1;
        if j == L0
            known = {copy(track{pair(1)}), copy(track{pair(2)})};  % models at the moment of loss
        end
        if j == L1  % back, no names -> compare positions
            for i = 1:2
                d = pair(i); seen(i) = at(d);
                for m = 1:2
                    t = track{pair(m)};
                    S = (squeeze(t.P(1, 1, :))' + t.r) * t.surprise;
                    Lp(i, m) = -0.5 * sum((z{d}(at(d), :) - t.x(1, :)) .^ 2 ./ S + log(S));
                end
            end
        end
        if j >= L1 && mod(j, 5) == 0
            Lb = zeros(2); Lm = zeros(2);
            fly = at(pair(1)) - seen(1) >= 15 && ~isempty(clf.n);
            if fly
                for i = 1:2
                    d = pair(i);
                    p = z{d}(max(seen(i), at(d) - WINDOW):at(d), :);
                    pr = clf.classify(features(p, DT));
                    Lb(i, :) = log(pr(pair));
                    Lm(i, :) = [miss(known{1}, p, DT), miss(known{2}, p, DT)];
                end
                Lb = sure(Lb, record.vedenje);
                Lm = sure(-0.5 * Lm ./ (min(Lm, [], 2) / (size(p, 1) - 10)), record.gibanje);
                later(end + 1) = struct("t", clock + WINDOW, "key", "vedenje", "f", [], "e", ~swapped(Lb));
                later(end + 1) = struct("t", clock + WINDOW, "key", "gibanje", "f", [], "e", ~swapped(Lm));
            end
            every = swapped(Lp + Lb + Lm);
            if every, who = fliplr(pair); else, who = pair; end
            nan_if = @(ok, v) ifelse(ok, v, NaN);
            ids(end + 1, :) = [clock * DT, ~every, ~swapped(Lp), nan_if(fly, ~swapped(Lb)), nan_if(fly, ~swapped(Lm))];
        end
        for i = 1:2
            d = pair(i);
            if lost
                track{pair(i)}.coast();
            else
                track{who(i)}.step(z{d}(at(d), :));
                flown{who(i)}(end + 1, :) = track{who(i)}.x(2, 1:2);
            end
        end
        while ~isempty(later) && later(1).t <= clock
            it = later(1); later(1) = [];
            if strlength(it.key)
                record.(it.key) = record.(it.key) + [it.e, 1];
            else
                clf.learn(it.f, it.f, it.e);
                windows(end + 1, :) = [it.f(c.FX), it.f(c.FY), it.e];
            end
        end
        if mod(j, 5) == 0
            state = struct("e", {}, "k", {}, "lost", {}, "guess", {}, "est", {}, "ours", {});
            for i = 1:2
                e = pair(i); k = at(e);
                ours = track{e}.predict(H); straight = track{e}.predict(H, false);
                if k > WINDOW && (j <= L0 || j >= L1 + WINDOW)
                    later(end + 1) = struct("t", clock + WINDOW, "key", "", "f", features(z{e}(k - WINDOW:k, :), DT), "e", e);
                end
                logm(end + 1, :) = [e, norm(ours(end, :) - flights{e}(k + H, :)), norm(straight(end, :) - flights{e}(k + H, :))];
                state(i) = struct("e", e, "k", k, "lost", lost, "guess", ifelse(lost, 0, who(i)), ...
                                  "est", track{e}.x(1, :), "ours", ours);
            end
            frames(end + 1) = struct("clock", clock, "state", state);
        end
        at(pair) = at(pair) + 1;
    end
end

%% ---- results + figures ----
fprintf("after the radar lost them, right name (2 in the air, chance 50 %%):\n");
what = ["position only", "behaviour only", "travel model only", "all together"];
cols = [3, 4, 5, 2];
for i = 1:4
    fprintf("  %-21s %3.0f %%\n", what(i), 100 * mean(ids(:, cols(i)), "omitnan"));
end
for e = 1:ne
    m = logm(:, 1) == e;
    fprintf("  %-9s miss 4 s ahead: EvoKF %5.2f m, constant velocity %5.2f m\n", names(e), mean(logm(m, 2)), mean(logm(m, 3)));
end

here = fileparts(mfilename("fullpath"));
warning("off", "MATLAB:print:ContentTypeImageSuggested");
fig = figure("Color", "w", "Position", [100 100 800 450]);
accuracy(axes(fig), ids);
exportgraphics(fig, fullfile(here, "behaviour.png"), "Resolution", 150);
show = [find(names == "mavic2"), find(names == "avata")];
seminar(here, frames, flights, names, show, steps / 5, clf, track, windows, flown, DT);
end


%% ---- helpers ----

function f = features(p, DT)
v = diff(p) / DT;
a = diff(v) / DT;
v = v(2:end, :);
s = hypot(v(:, 1), v(:, 2));
along = (v(:, 1) .* a(:, 1) + v(:, 2) .* a(:, 2)) ./ max(s, 1e-6);
side = abs(v(:, 1) .* a(:, 2) - v(:, 2) .* a(:, 1)) ./ max(s, 1e-6);
jerk = vecnorm(diff(a(:, 1:2)), 2, 2) / DT;
mv = s > 0.5;
if sum(mv) <= 2, mv = true(size(s)); end
f = [mean(s), pct(s, 90), mean(max(along(mv), 0)), mean(max(-along(mv), 0)), ...
     mean(side(mv)), pct(side(mv), 90), pct(side(mv) ./ max(s(mv), 0.5) .^ 2, 90), median(jerk), ...
     pct(abs(v(:, 3)), 90), 1 - mean(s > 0.5)];
end

function q = pct(x, p)
x = sort(x(:)); r = (numel(x) - 1) * p / 100; i = floor(r);
q = x(i + 1) + (r - i) * (x(min(i + 2, end)) - x(i + 1));
end

function m = miss(model, p, DT)
v = (p(6:end, :) - p(1:end - 5, :)) / (5 * DT); m = 0;
for i = 1:size(v, 1) - 5
    a = v(i, :);
    for s = 1:5
        if ~isempty(model.n), a = model.next(a); end
    end
    m = m + sum((a - v(i + 5, :)) .^ 2);
end
end

function L = sure(L, right)
c = min(max(right(1) / right(2), 0.5), 0.95);
p = exp(L - max(L, [], 2));
L = log((2 * c - 1) * p ./ sum(p, 2) + 1 - c);
end

function s = swapped(L)
s = L(1, 2) + L(2, 1) > L(1, 1) + L(2, 2);
end

function R = turn(i)
a = 2 * pi * (i - 1) / 5;
R = [cos(a), sin(a), 0; -sin(a), cos(a), 0; 0, 0, 1];
end

function out = ifelse(cond, a, b)
if cond, out = a; else, out = b; end
end

function c = constants()
c.COLORS = [42 120 214; 235 104 52; 27 175 122; 237 161 0; 232 123 164] / 255;
c.FEATURES = ["hitrost", "najvišja hitrost", "pospeševanje", "zaviranje", "bočni pospešek", ...
              "največji bočni pospešek", "ostrina zavoja", "sunkovitost", "dviganje", "lebdenje"];
c.FX = 4; c.FY = 6;
end


%% ---- figures ----

function txt = special(clf, c, top)
k = constants(); txt = "";
if clf.cn(c) < 2, return, end
S = clf.special(); S = S(c, :);
rest = setdiff(1:clf.nclass, c); others = clf.cm(rest(clf.cn(rest) > 1), :);
[~, o] = sort(S, "descend");
parts = strings(1, top);
for i = 1:top
    f = o(i);
    if clf.cm(c, f) > median(others(:, f)), arrow = char(8593); else, arrow = char(8595); end
    parts(i) = arrow + " " + k.FEATURES(f);
end
txt = strjoin(parts, ", ");
end

function [mu, S, go] = how_roji(track, DT)
mu = track.mu(:, 1:2); S = track.covariance(); S = S(1:2, 1:2, :);
go = zeros(size(mu));
for r = 1:numel(track.n)
    theta = track.A(:, :, r) \ track.B(:, :, r);
    change = (theta(2:end, :)' * track.mu(r, :)')' + theta(1, :);
    go(r, :) = mu(r, :) + change(1:2) / DT;
end
end

function roj(ax, c, S, color, lw)
xy = ellipse(c, S);
fill(ax, xy(1, :), xy(2, :), color, "FaceAlpha", 0.15, "EdgeColor", color, "LineWidth", lw, "HandleVisibility", "off");
end


function xy = ellipse(c, S)
t = linspace(0, 2 * pi, 60);
[V, D] = eig(S + eye(2) * 0.03 ^ 2);
xy = c(:) + 1.5 * V * (sqrt(max(diag(D), 0)) .* [cos(t); sin(t)]);
end

function view3d(a)
view(a, 30, 25);
pbaspect(a, [1.5 1.1 0.75]);
end

function accuracy(a, ids)
c = constants(); cla(a); hold(a, "on");
if ~isempty(ids)
    secs = unique(ids(:, 1));
    acc = zeros(numel(secs), 4);
    for i = 1:numel(secs)
        m = ids(:, 1) > secs(i) - 60 & ids(:, 1) <= secs(i);
        acc(i, :) = 100 * mean(ids(m, 2:5), 1, "omitnan");
    end
    plot(a, secs, acc(:, 1), "-", "Color", c.COLORS(1, :), "LineWidth", 2, "DisplayName", "vse skupaj");
    plot(a, secs, acc(:, 2), "--", "Color", [.5 .5 .5], "LineWidth", 1.2, "DisplayName", "samo napovedan položaj");
    plot(a, secs, acc(:, 3), ":", "Color", c.COLORS(2, :), "LineWidth", 1.2, "DisplayName", "samo vedenje");
    plot(a, secs, acc(:, 4), "-.", "Color", c.COLORS(3, :), "LineWidth", 1.2, "DisplayName", "samo model gibanja");
    legend(a, "Location", "southeast", "Box", "off");
end
yline(a, 50, ":", "Color", [.5 .5 .5], "HandleVisibility", "off");
ylim(a, [0 105]); xlabel(a, "čas [s]"); ylabel(a, "pravo ime po izgubi, zadnjih 60 s [%]");
end

function seminar(folder, frames, flights, names, show, per, clf, track, windows, flown, DT)
c = constants();
fig = figure("Color", "w", "Units", "inches", "Position", [1 1 6.3 4.2], "Visible", "off");
a = axes(fig); hold(a, "on"); grid(a, "on"); view3d(a);
for e = show
    q = flights{e};
    mine = [];
    for j = 1:numel(frames)
        for s = frames(j).state
            if s.e == e, mine(end + 1, :) = [j, s.k, s.lost, s.est, s.ours(end, :)]; end
        end
    end
    plot3(a, q(min(mine(:, 2)):max(mine(:, 2)), 1), q(min(mine(:, 2)):max(mine(:, 2)), 2), q(min(mine(:, 2)):max(mine(:, 2)), 3), ...
          "-", "Color", c.COLORS(e, :), "LineWidth", 2, "DisplayName", names(e));
    for s = unique(floor((mine(:, 1) - 1) / per))'
        seg = mine(floor((mine(:, 1) - 1) / per) == s, :);
        gap = seg(seg(:, 3) == 1, :);
        if isempty(gap), continue, end
        g0 = gap(1, 2); g1 = gap(end, 2) + 5;
        plot3(a, q(g0:g1, 1), q(g0:g1, 2), q(g0:g1, 3), "-", "Color", "w", "LineWidth", 3.5, "HandleVisibility", "off");
        plot3(a, q(g0:g1, 1), q(g0:g1, 2), q(g0:g1, 3), ":", "Color", c.COLORS(e, :), "LineWidth", 2.5, "HandleVisibility", "off");
        pr = [gap(1, 4:6); track_prediction(frames, gap(1, 1), e)];
        plot3(a, pr(:, 1), pr(:, 2), pr(:, 3), "--k", "LineWidth", 2, "HandleVisibility", "off");
        u = (q(g0, :) - q(g0 - 3, :)) / max(norm(q(g0, :) - q(g0 - 3, :)), 1e-9);
        quiver3(a, q(g0, 1) - 0.9 * u(1), q(g0, 2) - 0.9 * u(2), q(g0, 3) - 0.9 * u(3), 0.9 * u(1), 0.9 * u(2), 0.9 * u(3), 0, ...
                "Color", c.COLORS(e, :), "LineWidth", 2.5, "MaxHeadSize", 3, "HandleVisibility", "off");
        plot3(a, q(g1, 1), q(g1, 2), q(g1, 3), "o", "MarkerSize", 8, "MarkerFaceColor", c.COLORS(e, :), ...
              "MarkerEdgeColor", c.COLORS(e, :), "HandleVisibility", "off");
    end
end
plot3(a, NaN, NaN, NaN, ":", "Color", [.5 .5 .5], "LineWidth", 2.5, "DisplayName", "senzor ga ne vidi");
plot3(a, NaN, NaN, NaN, "--k", "LineWidth", 2, "DisplayName", "napoved EvoKF");
plot3(a, NaN, NaN, NaN, ">", "Color", [.5 .5 .5], "MarkerFaceColor", [.5 .5 .5], "DisplayName", "tu ga izgubi");
plot3(a, NaN, NaN, NaN, "o", "Color", [.5 .5 .5], "MarkerFaceColor", [.5 .5 .5], "DisplayName", "spet najden");
zticks(a, [0 5 10 15]); xlabel(a, "x [m]"); ylabel(a, "y [m]"); zlabel(a, "višina [m]");
legend(a, "Location", "northwest", "Box", "off", "NumColumns", 2, "FontSize", 8);
exportgraphics(fig, fullfile(folder, "seminar_sledenje.pdf"), "ContentType", "vector");

fig = figure("Color", "w", "Units", "inches", "Position", [1 1 4.2 4.2], "Visible", "off");
a = axes(fig); hold(a, "on"); box(a, "on");
scatter(a, windows(:, 1), windows(:, 2), 10, c.COLORS(windows(:, 3), :), "filled", "MarkerFaceAlpha", 0.4, "HandleVisibility", "off");
S = clf.covariance();
for r = 1:numel(clf.n)
    [~, w] = max(clf.L(r, :));
    roj(a, clf.mu(r, [c.FX c.FY]), S([c.FX c.FY], [c.FX c.FY], r), c.COLORS(w, :), 2.2);
end
for e = 1:numel(names)
    plot(a, NaN, NaN, "-", "Color", c.COLORS(e, :), "LineWidth", 2.5, "DisplayName", names(e) + ": " + special(clf, e, 1));
end
yl = ylim(a); ylim(a, [yl(1), yl(2) + 0.6 * diff(yl)]);
xlabel(a, c.FEATURES(c.FX) + " [m/s^2]"); ylabel(a, c.FEATURES(c.FY) + " [m/s^2]");
legend(a, "Location", "northwest", "Box", "off", "NumColumns", 2, "FontSize", 8);
exportgraphics(fig, fullfile(folder, "seminar_razvrscevalnik.pdf"), "ContentType", "vector");

fig = figure("Color", "w", "Units", "inches", "Position", [1 1 4.2 4.2], "Visible", "off");
a = axes(fig); hold(a, "on"); box(a, "on"); axis(a, "equal");
big = 0;
for e = 1:numel(names)
    R = turn(e); R = R(1:2, 1:2)';
    f = flown{e} * R;
    scatter(a, f(:, 1), f(:, 2), 5, c.COLORS(e, :), "filled", "MarkerFaceAlpha", 0.25, "HandleVisibility", "off");
    [mu, Sr, go] = how_roji(track{e}, DT);
    mu = mu * R; go = go * R;
    for r = 1:size(mu, 1)
        Srot = R' * Sr(:, :, r) * R;
        roj(a, mu(r, :), Srot, c.COLORS(e, :), 2.2);
        quiver(a, mu(r, 1), mu(r, 2), go(r, 1) - mu(r, 1), go(r, 2) - mu(r, 2), 0, "Color", c.COLORS(e, :), ...
               "LineWidth", 2, "MaxHeadSize", 0.8, "HandleVisibility", "off");
        big = max(big, max(abs(mu(r, :))) + 1.5 * sqrt(max(eig(Srot))));
    end
    big = max(big, pct(abs(f(:)), 99.5));
end
for e = 1:numel(names)
    plot(a, NaN, NaN, "-", "Color", c.COLORS(e, :), "LineWidth", 2.5, "DisplayName", names(e));
end
xlim(a, 1.05 * [-big big]); ylim(a, 1.05 * [-big big]);
xline(a, 0, "Color", [.6 .6 .6], "HandleVisibility", "off"); yline(a, 0, "Color", [.6 .6 .6], "HandleVisibility", "off");
xlabel(a, "hitrost x [m/s]"); ylabel(a, "hitrost y [m/s]");
legend(a, "Location", "northwest", "Box", "off", "FontSize", 8);
exportgraphics(fig, fullfile(folder, "seminar_regresija.pdf"), "ContentType", "vector");
end

function pr = track_prediction(frames, j, e)
for s = frames(j).state
    if s.e == e, pr = s.ours; return, end
end
end
