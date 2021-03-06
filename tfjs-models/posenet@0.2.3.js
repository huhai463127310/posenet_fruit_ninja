// @tensorflow/tfjs-models Copyright 2018 Google
! function (e, t) {
    "object" == typeof exports && "undefined" != typeof module ? t(exports, require("@tensorflow/tfjs")) : "function" == typeof define && define.amd ? define(["exports", "@tensorflow/tfjs"], t) : t(e.posenet = {}, e.tf)
}(this, function (e, t) {
    "use strict";
    var n = function () {
            function e(e) {
                this.urlPath = e, "/" !== this.urlPath.charAt(this.urlPath.length - 1) && (this.urlPath += "/")
            }
            return e.prototype.loadManifest = function () {
                var e = this;
                return new Promise(function (t, n) {
                    var r = new XMLHttpRequest;
                    r.open("GET", e.urlPath + "manifest.json"), r.onload = function () {
                        e.checkpointManifest = JSON.parse(r.responseText), t()
                    }, r.onerror = function (t) {
                        throw new Error("manifest.json not found at " + e.urlPath + ". " + t)
                    }, r.send()
                })
            }, e.prototype.getCheckpointManifest = function () {
                var e = this;
                return null == this.checkpointManifest ? new Promise(function (t, n) {
                    e.loadManifest().then(function () {
                        t(e.checkpointManifest)
                    })
                }) : new Promise(function (t, n) {
                    t(e.checkpointManifest)
                })
            }, e.prototype.getAllVariables = function () {
                var e = this;
                return null != this.variables ? new Promise(function (t, n) {
                    t(e.variables)
                }) : new Promise(function (t, n) {
                    e.getCheckpointManifest().then(function (n) {
                        for (var r = Object.keys(e.checkpointManifest), i = [], o = 0; o < r.length; o++) i.push(e.getVariable(r[o]));
                        Promise.all(i).then(function (n) {
                            e.variables = {};
                            for (var i = 0; i < n.length; i++) e.variables[r[i]] = n[i];
                            t(e.variables)
                        })
                    })
                })
            }, e.prototype.getVariable = function (e) {
                var n = this;
                if (!(e in this.checkpointManifest)) throw new Error("Cannot load non-existant variable " + e);
                var r = function (r, i) {
                    var o = new XMLHttpRequest;
                    o.responseType = "arraybuffer";
                    var a = n.checkpointManifest[e].filename;
                    o.open("GET", n.urlPath + a), o.onload = function () {
                        if (404 === o.status) throw new Error("Not found variable " + e);
                        var i = new Float32Array(o.response),
                            a = t.Tensor.make(n.checkpointManifest[e].shape, {
                                values: i
                            });
                        r(a)
                    }, o.onerror = function (t) {
                        throw new Error("Could not fetch variable " + e + ": " + t)
                    }, o.send()
                };
                return null == this.checkpointManifest ? new Promise(function (e, t) {
                    n.loadManifest().then(function () {
                        new Promise(r).then(e)
                    })
                }) : new Promise(r)
            }, e
        }(),
        r = [8, 16, 32];

    function i(e) {
        t.util.assert("number" == typeof e, "outputStride is not a number"), t.util.assert(r.indexOf(e) >= 0, "outputStride of " + e + " is invalid. It must be either 8, 16, or 32")
    }

    function o(e) {
        t.util.assert("number" == typeof e, "imageScaleFactor is not a number"), t.util.assert(e >= .2 && e <= 1, "imageScaleFactor must be between 0.2 and 1.0")
    }
    var a = {
        100: [
            ["conv2d", 2],
            ["separableConv", 1],
            ["separableConv", 2],
            ["separableConv", 1],
            ["separableConv", 2],
            ["separableConv", 1],
            ["separableConv", 2],
            ["separableConv", 1],
            ["separableConv", 1],
            ["separableConv", 1],
            ["separableConv", 1],
            ["separableConv", 1],
            ["separableConv", 2],
            ["separableConv", 1]
        ],
        75: [
            ["conv2d", 2],
            ["separableConv", 1],
            ["separableConv", 2],
            ["separableConv", 1],
            ["separableConv", 2],
            ["separableConv", 1],
            ["separableConv", 2],
            ["separableConv", 1],
            ["separableConv", 1],
            ["separableConv", 1],
            ["separableConv", 1],
            ["separableConv", 1],
            ["separableConv", 1],
            ["separableConv", 1]
        ],
        50: [
            ["conv2d", 2],
            ["separableConv", 1],
            ["separableConv", 2],
            ["separableConv", 1],
            ["separableConv", 2],
            ["separableConv", 1],
            ["separableConv", 2],
            ["separableConv", 1],
            ["separableConv", 1],
            ["separableConv", 1],
            ["separableConv", 1],
            ["separableConv", 1],
            ["separableConv", 1],
            ["separableConv", 1]
        ]
    };
    var s = function () {
        function e(e, n) {
            this.PREPROCESS_DIVISOR = t.scalar(127.5), this.ONE = t.scalar(1), this.variables = e, this.convolutionDefinitions = n
        }
        return e.prototype.predict = function (e, n) {
            var r = this,
                i = t.cast(e, "float32").div(this.PREPROCESS_DIVISOR).sub(this.ONE);
            return function (e, t) {
                var n = 1,
                    r = 1;
                return e.map(function (e, i) {
                    var o, a, s = e[0],
                        u = e[1];
                    return n === t ? (o = 1, a = r, r *= u) : (o = u, a = 1, n *= u), {
                        blockId: i,
                        convType: s,
                        stride: o,
                        rate: a,
                        outputStride: n
                    }
                })
            }(this.convolutionDefinitions, n).reduce(function (e, t) {
                var n = t.blockId,
                    i = t.stride,
                    o = t.convType,
                    a = t.rate;
                if ("conv2d" === o) return r.conv(e, i, n);
                if ("separableConv" === o) return r.separableConv(e, i, n, a);
                throw Error("Unknown conv type of " + o)
            }, i)
        }, e.prototype.convToOutput = function (e, t) {
            return e.conv2d(this.weights(t), 1, "same").add(this.biases(t))
        }, e.prototype.conv = function (e, t, n) {
            return e.conv2d(this.weights("Conv2d_" + String(n)), t, "same").add(this.biases("Conv2d_" + String(n))).clipByValue(0, 6)
        }, e.prototype.separableConv = function (e, t, n, r) {
            void 0 === r && (r = 1);
            var i = "Conv2d_" + String(n) + "_depthwise",
                o = "Conv2d_" + String(n) + "_pointwise";
            return e.depthwiseConv2D(this.depthwiseWeights(i), t, "same", "NHWC", r).add(this.biases(i)).clipByValue(0, 6).conv2d(this.weights(o), [1, 1], "same").add(this.biases(o)).clipByValue(0, 6)
        }, e.prototype.weights = function (e) {
            return this.variables["MobilenetV1/" + e + "/weights"]
        }, e.prototype.biases = function (e) {
            return this.variables["MobilenetV1/" + e + "/biases"]
        }, e.prototype.depthwiseWeights = function (e) {
            return this.variables["MobilenetV1/" + e + "/depthwise_weights"]
        }, e.prototype.dispose = function () {
            for (var e in this.variables) this.variables[e].dispose()
        }, e
    }();

    function u(e, t, n, r) {
        return new(n || (n = Promise))(function (i, o) {
            function a(e) {
                try {
                    u(r.next(e))
                } catch (e) {
                    o(e)
                }
            }

            function s(e) {
                try {
                    u(r.throw(e))
                } catch (e) {
                    o(e)
                }
            }

            function u(e) {
                e.done ? i(e.value) : new n(function (t) {
                    t(e.value)
                }).then(a, s)
            }
            u((r = r.apply(e, t || [])).next())
        })
    }

    function l(e, t) {
        var n, r, i, o, a = {
            label: 0,
            sent: function () {
                if (1 & i[0]) throw i[1];
                return i[1]
            },
            trys: [],
            ops: []
        };
        return o = {
            next: s(0),
            throw: s(1),
            return: s(2)
        }, "function" == typeof Symbol && (o[Symbol.iterator] = function () {
            return this
        }), o;

        function s(o) {
            return function (s) {
                return function (o) {
                    if (n) throw new TypeError("Generator is already executing.");
                    for (; a;) try {
                        if (n = 1, r && (i = 2 & o[0] ? r.return : o[0] ? r.throw || ((i = r.return) && i.call(r), 0) : r.next) && !(i = i.call(r, o[1])).done) return i;
                        switch (r = 0, i && (o = [2 & o[0], i.value]), o[0]) {
                            case 0:
                            case 1:
                                i = o;
                                break;
                            case 4:
                                return a.label++, {
                                    value: o[1],
                                    done: !1
                                };
                            case 5:
                                a.label++, r = o[1], o = [0];
                                continue;
                            case 7:
                                o = a.ops.pop(), a.trys.pop();
                                continue;
                            default:
                                if (!(i = (i = a.trys).length > 0 && i[i.length - 1]) && (6 === o[0] || 2 === o[0])) {
                                    a = 0;
                                    continue
                                }
                                if (3 === o[0] && (!i || o[1] > i[0] && o[1] < i[3])) {
                                    a.label = o[1];
                                    break
                                }
                                if (6 === o[0] && a.label < i[1]) {
                                    a.label = i[1], i = o;
                                    break
                                }
                                if (i && a.label < i[2]) {
                                    a.label = i[2], a.ops.push(o);
                                    break
                                }
                                i[2] && a.ops.pop(), a.trys.pop();
                                continue
                        }
                        o = t.call(e, a)
                    } catch (e) {
                        o = [6, e], r = 0
                    } finally {
                        n = i = 0
                    }
                    if (5 & o[0]) throw o[1];
                    return {
                        value: o[0] ? o[1] : void 0,
                        done: !0
                    }
                }([o, s])
            }
        }
    }
    var c = ["nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder", "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist", "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"],
        p = c.length,
        f = c.reduce(function (e, t, n) {
            return e[t] = n, e
        }, {}),
        h = [
            ["nose", "leftEye"],
            ["leftEye", "leftEar"],
            ["nose", "rightEye"],
            ["rightEye", "rightEar"],
            ["nose", "leftShoulder"],
            ["leftShoulder", "leftElbow"],
            ["leftElbow", "leftWrist"],
            ["leftShoulder", "leftHip"],
            ["leftHip", "leftKnee"],
            ["leftKnee", "leftAnkle"],
            ["nose", "rightShoulder"],
            ["rightShoulder", "rightElbow"],
            ["rightElbow", "rightWrist"],
            ["rightShoulder", "rightHip"],
            ["rightHip", "rightKnee"],
            ["rightKnee", "rightAnkle"]
        ],
        v = [
            ["leftHip", "leftShoulder"],
            ["leftElbow", "leftShoulder"],
            ["leftElbow", "leftWrist"],
            ["leftHip", "leftKnee"],
            ["leftKnee", "leftAnkle"],
            ["rightHip", "rightShoulder"],
            ["rightElbow", "rightShoulder"],
            ["rightElbow", "rightWrist"],
            ["rightHip", "rightKnee"],
            ["rightKnee", "rightAnkle"],
            ["leftShoulder", "rightShoulder"],
            ["leftHip", "rightHip"]
        ].map(function (e) {
            var t = e[0],
                n = e[1];
            return [f[t], f[n]]
        });
    var d = Number.NEGATIVE_INFINITY,
        b = Number.POSITIVE_INFINITY;

    function m(e) {
        return e.reduce(function (e, t) {
            var n = e.maxX,
                r = e.maxY,
                i = e.minX,
                o = e.minY,
                a = t.position,
                s = a.x,
                u = a.y;
            return {
                maxX: Math.max(n, s),
                maxY: Math.max(r, u),
                minX: Math.min(i, s),
                minY: Math.min(o, u)
            }
        }, {
            maxX: d,
            maxY: d,
            minX: b,
            minY: b
        })
    }

    function y(e, n) {
        return void 0 === n && (n = "float32"), u(this, void 0, void 0, function () {
            var r;
            return l(this, function (i) {
                switch (i.label) {
                    case 0:
                        return [4, e.data()];
                    case 1:
                        return r = i.sent(), [2, new t.TensorBuffer(e.shape, n, r)]
                }
            })
        })
    }

    function g(e, t, n) {
        return {
            score: e.score,
            keypoints: e.keypoints.map(function (e) {
                var r = e.score,
                    i = e.part,
                    o = e.position;
                return {
                    score: r,
                    part: i,
                    position: {
                        x: o.x * t,
                        y: o.y * n
                    }
                }
            })
        }
    }

    function w(e, t, n) {
        var r = t * e - 1;
        return r - r % n + 1
    }

    function x(e) {
        return Math.floor(e / 2)
    }
    var C = function () {
        function e(e, t) {
            this.priorityQueue = new Array(e), this.numberOfElements = -1, this.getElementValue = t
        }
        return e.prototype.enqueue = function (e) {
            this.priorityQueue[++this.numberOfElements] = e, this.swim(this.numberOfElements)
        }, e.prototype.dequeue = function () {
            var e = this.priorityQueue[0];
            return this.exchange(0, this.numberOfElements--), this.sink(0), this.priorityQueue[this.numberOfElements + 1] = null, e
        }, e.prototype.empty = function () {
            return -1 === this.numberOfElements
        }, e.prototype.size = function () {
            return this.numberOfElements + 1
        }, e.prototype.all = function () {
            return this.priorityQueue.slice(0, this.numberOfElements + 1)
        }, e.prototype.max = function () {
            return this.priorityQueue[0]
        }, e.prototype.swim = function (e) {
            for (; e > 0 && this.less(x(e), e);) this.exchange(e, x(e)), e = x(e)
        }, e.prototype.sink = function (e) {
            for (; 2 * e <= this.numberOfElements;) {
                var t = 2 * e;
                if (t < this.numberOfElements && this.less(t, t + 1) && t++, !this.less(e, t)) break;
                this.exchange(e, t), e = t
            }
        }, e.prototype.getValueAt = function (e) {
            return this.getElementValue(this.priorityQueue[e])
        }, e.prototype.less = function (e, t) {
            return this.getValueAt(e) < this.getValueAt(t)
        }, e.prototype.exchange = function (e, t) {
            var n = this.priorityQueue[e];
            this.priorityQueue[e] = this.priorityQueue[t], this.priorityQueue[t] = n
        }, e
    }();

    function E(e, t, n, r, i, o) {
        for (var a = o.shape, s = a[0], u = a[1], l = !0, c = Math.max(n - i, 0), p = Math.min(n + i + 1, s), f = c; f < p; ++f) {
            for (var h = Math.max(r - i, 0), v = Math.min(r + i + 1, u), d = h; d < v; ++d)
                if (o.get(f, d, e) > t) {
                    l = !1;
                    break
                } if (!l) break
        }
        return l
    }

    function S(e, t, n, r) {
        return {
            y: r.get(e, t, n),
            x: r.get(e, t, n + p)
        }
    }

    function k(e, t, n) {
        var r = S(e.heatmapY, e.heatmapX, e.id, n),
            i = r.y,
            o = r.x;
        return {
            x: e.heatmapX * t + o,
            y: e.heatmapY * t + i
        }
    }

    function M(e, t, n) {
        return e < t ? t : e > n ? n : e
    }

    function P(e, t) {
        return {
            x: e.x + t.x,
            y: e.y + t.y
        }
    }
    var O = h.map(function (e) {
            var t = e[0],
                n = e[1];
            return [f[t], f[n]]
        }),
        _ = O.map(function (e) {
            return e[1]
        }),
        N = O.map(function (e) {
            return e[0]
        });

    function T(e, t, n, r) {
        return {
            y: M(Math.round(e.y / t), 0, n - 1),
            x: M(Math.round(e.x / t), 0, r - 1)
        }
    }

    function A(e, t, n, r, i, o, a) {
        var s = r.shape,
            u = s[0],
            l = s[1],
            p = function (e, t, n) {
                var r = n.shape[2] / 2;
                return {
                    y: n.get(t.y, t.x, e),
                    x: n.get(t.y, t.x, r + e)
                }
            }(e, T(t.position, o, u, l), a),
            f = T(P(t.position, p), o, u, l),
            h = S(f.y, f.x, n, i),
            v = r.get(f.y, f.x, n);
        return {
            position: P({
                x: f.x * o,
                y: f.y * o
            }, {
                x: h.x,
                y: h.y
            }),
            part: c[n],
            score: v
        }
    }

    function V(e, t, n, r, i, o) {
        var a = t.shape[2],
            s = _.length,
            u = new Array(a),
            l = e.part,
            p = e.score,
            f = k(l, r, n);
        u[l.id] = {
            score: p,
            part: c[l.id],
            position: f
        };
        for (var h = s - 1; h >= 0; --h) {
            var v = _[h],
                d = N[h];
            u[v] && !u[d] && (u[d] = A(h, u[v], d, t, n, r, o))
        }
        for (h = 0; h < s; ++h) {
            v = N[h], d = _[h];
            u[v] && !u[d] && (u[d] = A(h, u[v], d, t, n, r, i))
        }
        return u
    }

    function I(e, t, n, r) {
        var i = n.x,
            o = n.y;
        return e.some(function (e) {
            var n, a, s, u, l, c, p = e.keypoints[r].position;
            return n = o, a = i, s = p.y, u = p.x, (l = s - n) * l + (c = u - a) * c <= t
        })
    }

    function H(e, t, n) {
        return n.reduce(function (n, r, i) {
            var o = r.position,
                a = r.score;
            return I(e, t, o, i) || (n += a), n
        }, 0) / n.length
    }
    var F = 1;

    function j(e, t, n, r, i, o, a, s) {
        return void 0 === a && (a = .5), void 0 === s && (s = 20), u(this, void 0, void 0, function () {
            var c, p, f, h, v, d, b, m, g, w, x, S;
            return l(this, function (M) {
                switch (M.label) {
                    case 0:
                        return c = [], [4, function (e) {
                            return u(this, void 0, void 0, function () {
                                return l(this, function (t) {
                                    return [2, Promise.all(e.map(function (e) {
                                        return y(e, "float32")
                                    }))]
                                })
                            })
                        }([e, t, n, r])];
                    case 1:
                        for (p = M.sent(), f = p[0], h = p[1], v = p[2], d = p[3], b = function (e, t, n) {
                                for (var r = n.shape, i = r[0], o = r[1], a = r[2], s = new C(i * o * a, function (e) {
                                        return e.score
                                    }), u = 0; u < i; ++u)
                                    for (var l = 0; l < o; ++l)
                                        for (var c = 0; c < a; ++c) {
                                            var p = n.get(u, l, c);
                                            p < e || E(c, p, u, l, t, n) && s.enqueue({
                                                score: p,
                                                part: {
                                                    heatmapY: u,
                                                    heatmapX: l,
                                                    id: c
                                                }
                                            })
                                        }
                                return s
                            }(a, F, f), m = s * s; c.length < o && !b.empty();) g = b.dequeue(), w = k(g.part, i, h), I(c, m, w, g.part.id) || (x = V(g, f, h, i, v, d), S = H(c, m, x), c.push({
                            keypoints: x,
                            score: S
                        }));
                        return [2, c]
                }
            })
        })
    }
    // var X = "https://storage.googleapis.com/tfjs-models/weights/posenet/",
    var X = document.location.protocol + "//" + window.location.host + "/posenet_fruit_ninja/tfjs-models/weights/posenet/",
        Y = {
            1.01: {
                url: X + "mobilenet_v1_101/",
                architecture: a[100]
            },
            1: {
                url: X + "mobilenet_v1_100/",
                architecture: a[100]
            },
            .75: {
                url: X + "mobilenet_v1_075/",
                architecture: a[75]
            },
            .5: {
                url: X + "mobilenet_v1_050/",
                architecture: a[50]
            }
        };

    function B(e) {
        var n = e.shape,
            r = n[0],
            i = n[1],
            o = n[2];
        return t.tidy(function () {
            var n, a, s = e.reshape([r * i, o]).argMax(0),
                u = s.div(t.scalar(i, "int32")).expandDims(1),
                l = (n = s, a = i, t.tidy(function () {
                    var e = n.div(t.scalar(a, "int32"));
                    return n.sub(e.mul(t.scalar(a, "int32")))
                })).expandDims(1);
            return t.concat([u, l], 1)
        })
    }

    function K(e, t, n, r) {
        return {
            y: r.get(e, t, n),
            x: r.get(e, t, n + p)
        }
    }

    function Q(e, n, r) {
        return t.tidy(function () {
            var i = function (e, n) {
                for (var r = [], i = 0; i < p; i++) {
                    var o = K(e.get(i, 0).valueOf(), e.get(i, 1).valueOf(), i, n),
                        a = o.x,
                        s = o.y;
                    r.push(s), r.push(a)
                }
                return t.tensor2d(r, [p, 2])
            }(e, r);
            return e.toTensor().mul(t.scalar(n, "int32")).toFloat().add(i)
        })
    }

    function W(e, t, n) {
        return u(this, void 0, void 0, function () {
            var r, i, o, a, s, u, p, f, h, v;
            return l(this, function (l) {
                switch (l.label) {
                    case 0:
                        return r = 0, i = B(e), [4, Promise.all([y(e), y(t), y(i, "int32")])];
                    case 1:
                        return o = l.sent(), a = o[0], s = o[1], u = o[2], [4, y(p = Q(u, n, s))];
                    case 2:
                        return f = l.sent(), h = Array.from(function (e, t) {
                            for (var n = t.shape[0], r = new Float32Array(n), i = 0; i < n; i++) {
                                var o = t.get(i, 0),
                                    a = t.get(i, 1);
                                r[i] = e.get(o, a, i)
                            }
                            return r
                        }(a, u)), v = h.map(function (e, t) {
                            return r += e, {
                                position: {
                                    y: f.get(t, 0),
                                    x: f.get(t, 1)
                                },
                                part: c[t],
                                score: e
                            }
                        }), i.dispose(), p.dispose(), [2, {
                            keypoints: v,
                            score: r / v.length
                        }]
                }
            })
        })
    }

    function R(e, n, r, i) {
        var o = e instanceof t.Tensor ? e : t.fromPixels(e);
        return i ? o.reverse(1).resizeBilinear([n, r]) : o.resizeBilinear([n, r])
    }
    var q = function () {
        function e(e) {
            this.mobileNet = e
        }
        return e.prototype.predictForSinglePose = function (e, n) {
            var r = this;
            return void 0 === n && (n = 16), i(n), t.tidy(function () {
                var t = r.mobileNet.predict(e, n),
                    i = r.mobileNet.convToOutput(t, "heatmap_2"),
                    o = r.mobileNet.convToOutput(t, "offset_2");
                return {
                    heatmapScores: i.sigmoid(),
                    offsets: o
                }
            })
        }, e.prototype.predictForMultiPose = function (e, n) {
            var r = this;
            return void 0 === n && (n = 16), t.tidy(function () {
                var t = r.mobileNet.predict(e, n),
                    i = r.mobileNet.convToOutput(t, "heatmap_2"),
                    o = r.mobileNet.convToOutput(t, "offset_2"),
                    a = r.mobileNet.convToOutput(t, "displacement_fwd_2"),
                    s = r.mobileNet.convToOutput(t, "displacement_bwd_2");
                return {
                    heatmapScores: i.sigmoid(),
                    offsets: o,
                    displacementFwd: a,
                    displacementBwd: s
                }
            })
        }, e.prototype.estimateSinglePose = function (e, n, r, a) {
            return void 0 === n && (n = .5), void 0 === r && (r = !1), void 0 === a && (a = 16), u(this, void 0, void 0, function () {
                var s, u, c, p, f, h, v, d, b, m = this;
                return l(this, function (l) {
                    switch (l.label) {
                        case 0:
                            return i(a), o(n), s = e instanceof t.Tensor ? [e.shape[0], e.shape[1]] : [e.height, e.width], u = s[0], c = s[1], p = w(n, u, a), f = w(n, c, a), h = t.tidy(function () {
                                var t = R(e, p, f, r);
                                return m.predictForSinglePose(t, a)
                            }), v = h.heatmapScores, d = h.offsets, [4, W(v, d, a)];
                        case 1:
                            return b = l.sent(), v.dispose(), d.dispose(), [2, g(b, u / p, c / f)]
                    }
                })
            })
        }, e.prototype.estimateMultiplePoses = function (e, n, r, a, s, c, p) {
            return void 0 === n && (n = .5), void 0 === r && (r = !1), void 0 === a && (a = 16), void 0 === s && (s = 5), void 0 === c && (c = .5), void 0 === p && (p = 20), u(this, void 0, void 0, function () {
                var u, f, h, v, d, b, m, y, x, C, E, S = this;
                return l(this, function (l) {
                    switch (l.label) {
                        case 0:
                            return i(a), o(n), u = e instanceof t.Tensor ? [e.shape[0], e.shape[1]] : [e.height, e.width], f = u[0], h = u[1], v = w(n, f, a), d = w(n, h, a), b = t.tidy(function () {
                                var t = R(e, v, d, r);
                                return S.predictForMultiPose(t, a)
                            }), m = b.heatmapScores, y = b.offsets, x = b.displacementFwd, C = b.displacementBwd, [4, j(m, y, x, C, a, s, c, p)];
                        case 1:
                            return E = l.sent(), m.dispose(), y.dispose(), x.dispose(), C.dispose(), [2, function (e, t, n) {
                                return 1 === n && 1 === t ? e : e.map(function (e) {
                                    return g(e, n, t)
                                })
                            }(E, f / v, h / d)]
                    }
                })
            })
        }, e.prototype.dispose = function () {
            this.mobileNet.dispose()
        }, e
    }();
    var D = {
        load: function (e) {
            return u(void 0, void 0, void 0, function () {
                var t, r;
                return l(this, function (i) {
                    switch (i.label) {
                        case 0:
                            return t = Y[e], [4, new n(t.url).getAllVariables()];
                        case 1:
                            return r = i.sent(), [2, new s(r, t.architecture)]
                    }
                })
            })
        }
    };
    e.MobileNet = s, e.mobileNetArchitectures = a, e.CheckpointLoader = n, e.decodeMultiplePoses = j, e.decodeSinglePose = W, e.load = function (e) {
        return void 0 === e && (e = 1.01), u(this, void 0, void 0, function () {
            var n, r;
            return l(this, function (i) {
                switch (i.label) {
                    case 0:
                        if (null == t) throw new Error("Cannot find TensorFlow.js. If you are using a <script> tag, please also include @tensorflow/tfjs on the page before using this model.");
                        return n = Object.keys(Y), t.util.assert("number" == typeof e, "got multiplier type of " + typeof e + " when it should be a number."), t.util.assert(n.indexOf(e.toString()) >= 0, "invalid multiplier value of " + e + ".  No checkpoint exists for that multiplier. Must be one of " + n.join(",") + "."), [4, D.load(e)];
                    case 1:
                        return r = i.sent(), [2, new q(r)]
                }
            })
        })
    }, e.PoseNet = q, e.partIds = f, e.partNames = c, e.poseChain = h, e.getAdjacentKeyPoints = function (e, t) {
        return v.reduce(function (n, r) {
            var i = r[0],
                o = r[1];
            return function (e, t, n) {
                return e < n || t < n
            }(e[i].score, e[o].score, t) ? n : (n.push([e[i], e[o]]), n)
        }, [])
    }, e.getBoundingBox = m, e.getBoundingBoxPoints = function (e) {
        var t = m(e),
            n = t.minX,
            r = t.minY,
            i = t.maxX,
            o = t.maxY;
        return [{
            x: n,
            y: r
        }, {
            x: i,
            y: r
        }, {
            x: i,
            y: o
        }, {
            x: n,
            y: o
        }]
    }, Object.defineProperty(e, "__esModule", {
        value: !0
    })
});