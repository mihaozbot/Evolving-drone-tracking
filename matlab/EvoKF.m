% kalman on (p, v), swarms learn how v changes
classdef EvoKF < Swarms

    properties
        dt, r, F, Q
        x
        P
        steps = 0
        vPrev = []
        surprise = 1
    end

    methods
        function obj = EvoKF(dt, varargin)
            p = inputParser; p.KeepUnmatched = true;
            addParameter(p, 'q', 10.0); addParameter(p, 'r', 0.1 ^ 2);
            parse(p, varargin{:});
            rest = [fieldnames(p.Unmatched), struct2cell(p.Unmatched)]';
            obj@Swarms(rest{:});
            obj.dt = dt; obj.r = p.Results.r;
            obj.F = [1, dt; 0, 1];
            obj.Q = p.Results.q * [dt ^ 3 / 3, dt ^ 2 / 2; dt ^ 2 / 2, dt];
        end

        function start(obj, z0)
            obj.x = [z0; zeros(1, 3)];
            obj.P = repmat(diag([obj.r, 10]), 1, 1, 3);
            obj.steps = 0; obj.vPrev = []; obj.surprise = 1;
        end

        function step(obj, z, doLearn)
            if nargin < 3, doLearn = true; end
            obj.predictCovariance();
            obj.x = obj.F * obj.x;
            S = squeeze(obj.P(1, 1, :))' + obj.r;
            innovation = z - obj.x(1, :);
            obj.surprise = 0.95 * obj.surprise + 0.05 * mean(innovation .^ 2 ./ S);
            K = [squeeze(obj.P(1, 1, :))'; squeeze(obj.P(2, 1, :))'] ./ S;
            obj.x = obj.x + K .* innovation;
            for a = 1:3
                obj.P(:, :, a) = obj.P(:, :, a) - K(:, a) * obj.P(1, :, a);
            end
            obj.steps = obj.steps + 1;
            if doLearn && obj.steps > 5
                if ~isempty(obj.vPrev)
                    obj.learn(obj.vPrev, obj.x(2, :));
                end
                obj.vPrev = obj.x(2, :);
            end
        end

        function coast(obj)  % no measurement -> learned model
            if ~isempty(obj.n)
                obj.x(2, :) = obj.next(obj.x(2, :));
            end
            obj.x = obj.F * obj.x;
            obj.predictCovariance();
            obj.vPrev = [];
        end

        function path = predict(obj, H, learned)
            if nargin < 3, learned = true; end
            p = obj.x(1, :); v = obj.x(2, :); path = zeros(H, 3);
            for h = 1:H
                if learned && ~isempty(obj.n)
                    v = obj.next(v);
                end
                p = p + obj.dt * v;
                path(h, :) = p;
            end
        end
    end

    methods (Access = private)
        function predictCovariance(obj)
            for a = 1:3
                obj.P(:, :, a) = obj.F * obj.P(:, :, a) * obj.F' + obj.Q;
            end
        end
    end
end
