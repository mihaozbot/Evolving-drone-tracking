% swarms = local linear models, merge / drop to keep budget
classdef Swarms < matlab.mixin.Copyable

    properties
        budget, forget, ridge, dim, nclass
        L
        miss
        mu, n, last
        C
        A, B
        k = 0
        xm, xM2
        lo, hi
        cn, cm, cM2
    end

    methods
        function obj = Swarms(varargin)
            p = inputParser;
            addParameter(p, 'budget', 50); addParameter(p, 'forget', 2000);
            addParameter(p, 'ridge', 1e-2); addParameter(p, 'dim', 3); addParameter(p, 'classes', 0);
            parse(p, varargin{:});
            obj.budget = p.Results.budget; obj.forget = p.Results.forget; obj.ridge = p.Results.ridge;
            d = p.Results.dim; c = p.Results.classes;
            obj.dim = d; obj.nclass = c;
            obj.L = zeros(0, c); obj.miss = zeros(0, 1);
            obj.mu = zeros(0, d); obj.n = zeros(0, 1); obj.last = zeros(0, 1);
            obj.C = zeros(d, d, 0); obj.A = zeros(d + 1, d + 1, 0); obj.B = zeros(d + 1, d, 0);
            obj.xm = zeros(1, d); obj.xM2 = zeros(1, d);
            obj.lo = inf(1, d); obj.hi = -inf(1, d);
            obj.cn = zeros(c, 1); obj.cm = zeros(c, d); obj.cM2 = zeros(c, d);
        end

        function S = special(obj)
            S = ones(obj.nclass, obj.dim);
            seen = obj.cn > 1;
            if sum(seen) < 2, return, end
            for c = find(seen)'
                others = obj.cm(seen & (1:obj.nclass)' ~= c, :);
                S(c, :) = median((others - obj.cm(c, :)) .^ 2, 1) ./ (obj.cM2(c, :) / obj.cn(c) + 1e-12);
                S(c, :) = S(c, :) / mean(S(c, :)) + 1e-3;
            end
        end

        function W = weights(obj)
            if obj.nclass
                S = obj.special();
                [~, c] = max(obj.L, [], 2);
                W = S(c, :);
            else
                W = ones(size(obj.mu));
            end
        end

        function s = scale(obj)
            if obj.k > 1, s = sqrt(obj.xM2 / obj.k); else, s = ones(1, obj.dim); end
            s = max(s, 1e-9);
        end

        function d2 = dist2(obj, x)
            d2 = sum(((x - obj.mu) ./ obj.scale()) .^ 2 .* obj.weights(), 2);
        end

        %% ---- learn ----
        function learn(obj, x, y, label)
            if nargin < 4, label = []; end
            obj.k = obj.k + 1;
            y = y - x;  % learn the change
            obj.lo = min(obj.lo, y); obj.hi = max(obj.hi, y);
            d = x - obj.xm;
            obj.xm = obj.xm + d / obj.k;
            obj.xM2 = obj.xM2 + d .* (x - obj.xm);
            if ~isempty(obj.n)
                [~, near] = min(obj.dist2(x));
                obj.last(near) = obj.k;
            end
            if ~isempty(label)
                obj.cn(label) = obj.cn(label) + 1;
                d = x - obj.cm(label, :);
                obj.cm(label, :) = obj.cm(label, :) + d / obj.cn(label);
                obj.cM2(label, :) = obj.cM2(label, :) + d .* (x - obj.cm(label, :));
                if ~isempty(obj.n)
                    [~, c] = max(obj.L(near, :));
                    if c ~= label
                        obj.miss(near) = obj.miss(near) + 1;
                    end
                end
            end
            phi = [1, x];
            obj.mu(end + 1, :) = x; obj.n(end + 1, 1) = 1; obj.last(end + 1, 1) = obj.k;
            obj.C(:, :, end + 1) = zeros(obj.dim);
            obj.A(:, :, end + 1) = obj.ridge * eye(obj.dim + 1) + phi' * phi;
            obj.B(:, :, end + 1) = phi' * y;
            onehot = zeros(1, obj.nclass);
            if ~isempty(label), onehot(label) = 1; end
            obj.L(end + 1, :) = onehot; obj.miss(end + 1, 1) = 0;
            while numel(obj.n) > obj.budget
                obj.keepBudget();
            end
        end

        %% ---- budget: drop stale one or merge 2 closest ----
        function keepBudget(obj)
            X = obj.mu ./ obj.scale(); W = obj.weights(); N = size(X, 1);
            D = zeros(N);
            for i = 1:N
                D(i, :) = sum((X(i, :) - X) .^ 2 .* W(i, :), 2)';
            end
            D(1:N + 1:end) = inf;
            if obj.nclass  % only same class merges
                [~, c] = max(obj.L, [], 2);
                D(c ~= c') = inf;
                if ~any(isfinite(D(:)))
                    [~, i] = min(obj.n); obj.dropSwarm(i);
                    return
                end
            end
            nearest = min(D, [], 2);
            stale = find((obj.k - obj.last > obj.forget) & (nearest < median(nearest)));
            if ~isempty(stale)
                [~, i] = min(obj.last(stale)); obj.dropSwarm(stale(i));
                return
            end
            [~, idx] = min(D(:)); [a, b] = ind2sub(size(D), idx);
            if obj.n(a) >= obj.n(b), keep = a; gone = b; else, keep = b; gone = a; end
            w = obj.n(keep) * obj.n(gone) / (obj.n(keep) + obj.n(gone));
            d = obj.mu(keep, :) - obj.mu(gone, :);
            obj.C(:, :, keep) = obj.C(:, :, keep) + obj.C(:, :, gone) + w * (d' * d);
            obj.mu(keep, :) = (obj.n(keep) * obj.mu(keep, :) + obj.n(gone) * obj.mu(gone, :)) / (obj.n(keep) + obj.n(gone));
            obj.n(keep) = obj.n(keep) + obj.n(gone);
            obj.last(keep) = max(obj.last(keep), obj.last(gone));
            obj.A(:, :, keep) = obj.A(:, :, keep) + obj.A(:, :, gone) - obj.ridge * eye(obj.dim + 1);
            obj.B(:, :, keep) = obj.B(:, :, keep) + obj.B(:, :, gone);
            obj.L(keep, :) = obj.L(keep, :) + obj.L(gone, :);
            obj.miss(keep) = obj.miss(keep) + obj.miss(gone);
            obj.dropSwarm(gone);
        end

        function dropSwarm(obj, i)
            keep = true(numel(obj.n), 1); keep(i) = false;
            obj.mu = obj.mu(keep, :); obj.n = obj.n(keep); obj.last = obj.last(keep);
            obj.C = obj.C(:, :, keep); obj.A = obj.A(:, :, keep); obj.B = obj.B(:, :, keep);
            obj.L = obj.L(keep, :); obj.miss = obj.miss(keep);
        end

        function S = covariance(obj)
            S = obj.C ./ reshape(max(obj.n, 1), 1, 1, []);
        end

        %% ---- use ----
        function p = classify(obj, x, nNear)
            if nargin < 3, nNear = 8; end
            [o, w] = obj.nearest(x, nNear);
            trust = obj.n(o) ./ (obj.n(o) + obj.miss(o));  % often wrong -> trust less
            p = ((w .* trust)' * obj.L(o, :)) ./ max(sum(obj.L, 1), 1) + 1e-6;
        end

        function [Am, b] = motion(obj, x, nNear)
            if nargin < 3, nNear = 8; end
            [o, w] = obj.nearest(x, nNear);
            w = reshape(w, 1, 1, []);
            theta = (obj.ridge * eye(obj.dim + 1) + sum(obj.A(:, :, o) .* w, 3)) \ sum(obj.B(:, :, o) .* w, 3);
            Am = theta(2:end, :)'; b = theta(1, :);
        end

        function x = next(obj, x)
            [Am, b] = obj.motion(x);
            margin = 0.5 * (obj.hi - obj.lo);
            x = x + min(max((Am * x')' + b, obj.lo - margin), obj.hi + margin);
        end
    end

    methods (Access = private)
        function [o, w] = nearest(obj, x, nNear)
            d2 = obj.dist2(x);
            [~, o] = sort(d2);
            o = o(1:min(nNear, end));
            d = d2(o);
            w = exp(-(d - d(1)) / (median(d) - d(1) + 1e-9));
        end
    end
end
