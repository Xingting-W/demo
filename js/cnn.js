(function (global) {
    "use strict";

    var maxmin = cnnutil.maxmin;
    // draw_activations_COLOR
    var draw_activations_COLOR = function (elt, A, scale, grads) {
        var s = scale || 1;
        var draw_grads = false;
        if (typeof(grads) !== 'undefined') draw_grads = grads;

        // get max and min activation to scale the maps automatically
        var w = draw_grads ? A.dw : A.w;
        var mm = maxmin(w);

        var canv = document.createElement('canvas');
        canv.className = 'actmap';
        var W = A.sx * s;
        var H = A.sy * s;
        canv.width = W;
        canv.height = H;
        var ctx = canv.getContext('2d');
        var g = ctx.createImageData(W, H);
        for (var d = 0; d < 3; d++) {
            for (var x = 0; x < A.sx; x++) {
                for (var y = 0; y < A.sy; y++) {
                    if (draw_grads) {
                        var dval = Math.floor((A.get_grad(x, y, d) - mm.minv) / mm.dv * 255);
                    } else {
                        var dval = Math.floor((A.get(x, y, d) - mm.minv) / mm.dv * 255);
                    }
                    for (var dx = 0; dx < s; dx++) {
                        for (var dy = 0; dy < s; dy++) {
                            var pp = ((W * (y * s + dy)) + (dx + x * s)) * 4;
                            g.data[pp + d] = dval;
                            if (d === 0) g.data[pp + 3] = 255; // alpha channel
                        }
                    }
                }
            }
        }
        ctx.putImageData(g, 0, 0);
        elt.appendChild(canv);
    }

    var draw_activations = function(elt, A, scale, grads) {
        var s = scale || 1; // scale
        var draw_grads = false;
        if(typeof(grads) !== 'undefined') draw_grads = grads;

        // get max and min activation to scale the maps automatically
        var w = draw_grads ? A.dw : A.w;
        var mm = maxmin(w);

        // create the canvas elements, draw and add to DOM
        for(var d=0;d<A.depth;d++) {

            var canv = document.createElement('canvas');
            canv.className = 'actmap';
            var W = A.sx * s;
            var H = A.sy * s;
            canv.width = W;
            canv.height = H;
            var ctx = canv.getContext('2d');
            var g = ctx.createImageData(W, H);

            for(var x=0;x<A.sx;x++) {
                for(var y=0;y<A.sy;y++) {
                    if(draw_grads) {
                        var dval = Math.floor((A.get_grad(x,y,d)-mm.minv)/mm.dv*255);
                    } else {
                        var dval = Math.floor((A.get(x,y,d)-mm.minv)/mm.dv*255);
                    }
                    for(var dx=0;dx<s;dx++) {
                        for(var dy=0;dy<s;dy++) {
                            var pp = ((W * (y*s+dy)) + (dx + x*s)) * 4;
                            for(var i=0;i<3;i++) { g.data[pp + i] = dval; } // rgb
                            g.data[pp+3] = 255; // alpha channel
                        }
                    }
                }
            }
            ctx.putImageData(g, 0, 0);
            elt.appendChild(canv);
        }
    }

    global.draw_activations_COLOR = draw_activations_COLOR;
    global.draw_activations = draw_activations;

})(convnetjs);