$(document).ready(function() {
    var div = document.createElement('div'),
        canvas = document.createElement('canvas'),    
        ctx = canvas.getContext('2d'),
        w,
        h,
        msTimer = 0.0,
        lightningTimer,
        lightningAlpha,
        rainArr = [50],
        rainSpeed = 4;
  
    // initialize
    function init() {
      document.body.appendChild(div);
      div.style.position = "fixed";
      div.appendChild(canvas);
      UpdatePosition();
      create_rain();
      lightningTimer = 8000.0;
      lightningAlpha = 0.0;
  
      // 1 frame every 30ms
      if (typeof game_loop != "undefined") clearInterval(game_loop);
      game_loop = setInterval(mainLoop, 30);
    }
    init();
  
    function create_rain() {
      var length = 500;
      rainArr = []; //Empty array to start with
      for (var i = length - 1; i >= 0; i--) {
        rainArr.push({
          x: 1,
          y: 0,
          z: 0
        });
      }
  
      for (var j = 0; j < 500; j++) {
        rainArr[j].x = Math.floor((Math.random() * 820) - 9);
        rainArr[j].y = Math.floor((Math.random() * 520) - 9);
        rainArr[j].z = Math.floor((Math.random() * 2) + 1);
        rainArr[j].w = Math.floor((Math.random() * 3) + 2);
      }
    }
  
    function mainLoop() {
      UpdatePosition();
      msTimer += 30;
  
      if (lightningTimer < 0.0)  {
        lightningTimer = 8000.0;
      }
      else {
        lightningTimer -= 30.0;
      }    
  
      ctx.fillStyle = "#202426";
      ctx.fillRect(0,0,w,h);
  
      sidewalk();
      road();
      lamp();        
      rain();
  
      if (lightningTimer < 500.0) {
        weather(lightningTimer);
      }
  
      ctx.fillStyle = 'rgba(255, 255, 255, .1)';
      ctx.font = '30px Sans-Serif';
    }
    
    
    // canvas positioning and sizing
    function UpdatePosition () {
      var bodyWidth = document.documentElement.clientWidth,
          bodyHeight = document.documentElement.clientHeight;
      w = canvas.width = Math.max(500,bodyWidth);
      h = canvas.height = Math.max(320,bodyHeight);
      div.style.left=div.style.right=
        div.style.top=div.style.bottom="0";
    }
  
    // lamp visuals
    function lamp() {
      var grd = ctx.createLinearGradient(150, 210, 150, 500);
      grd.addColorStop(0.000, 'rgba(60, 60, 60, 1.000)');
      grd.addColorStop(0.2, 'rgba(80, 80, 80, 1.000)');
      grd.addColorStop(1, 'rgba(45, 45, 45, 1.000)');        
      ctx.fillStyle = grd;
      ctx.fillRect(247, 210, 6, 290);
  
      var grdOuterHigh = ctx.createLinearGradient(150, 210, 150, 500);        
      grdOuterHigh.addColorStop(0.000, 'rgba(65, 65, 65, 1.000)');
      grdOuterHigh.addColorStop(0.2, 'rgba(95, 95, 95, 1.000)');
      grdOuterHigh.addColorStop(1, 'rgba(47, 47, 47, 1.000)');
      ctx.fillStyle = grdOuterHigh;
      ctx.fillRect(246, 210, 1, 290);
  
      var grdOuterLow = ctx.createLinearGradient(150, 210, 150, 500);        
      grdOuterLow.addColorStop(0.000, 'rgba(45, 45, 45, 1.000)');
      grdOuterLow.addColorStop(0.2, 'rgba(60, 60, 60, 1.000)');
      grdOuterLow.addColorStop(1, 'rgba(43, 43, 43, 1.000)');
      ctx.fillStyle = grdOuterLow;
      ctx.fillRect(253, 210, 1, 290);
  
      // glow modified by time passed
      var sinGlowMod = 5 * Math.sin(msTimer / 200);
      var cosGlowMod = 5 * Math.cos((msTimer + 0.5 * sinGlowMod) / 200);        
      var grdGlow = ctx.createRadialGradient(250, 200, 0, 247 + sinGlowMod,
                                             400, 206 + cosGlowMod);
      grdGlow.addColorStop(0.000, 'rgba(220, 240, 160, 1)');
      grdGlow.addColorStop(0.2, 'rgba(180, 240, 160, 0.4)');
      grdGlow.addColorStop(0.4, 'rgba(140, 240, 160, 0.2)');
      grdGlow.addColorStop(1, 'rgba(140, 240, 160, 0)');
      ctx.fillStyle = grdGlow;
      ctx.fillRect(0, 0, 500, 500);
    }
  
    // function to position and color each rain drop
    // TODO: optimize - group raindrops together
    function rain() {   
      for (var i = 0; i < 500; i++) {
        if  (rainArr[i].y >= 482) {
          rainArr[i].y-=500;
        }
        if  (rainArr[i].x < -10) {
          rainArr[i].x+=w;
        }
        else {
          rainArr[i].y += rainArr[i].w * rainSpeed;
          rainArr[i].x -= 5 + Math.floor(rainArr[i].y / 250) - rainArr[i].w;
        }
  
        var grd = ctx.createRadialGradient(250, 450, 140, 250, 300, 600);
        grd.addColorStop(0.000, 'rgba(100, 170, 160, 0.2)');
        grd.addColorStop(0.1, 'rgba(100, 160, 160, 0.12)');
        grd.addColorStop(0.2, 'rgba(100, 150, 150, 0.1)');
        grd.addColorStop(1, 'rgba(100, 140, 140, .08)');
        ctx.fillStyle = grd;
        ctx.fillRect(rainArr[i].x, rainArr[i].y, rainArr[i].z, 4);
      }
    }
    
    // sidewalk visuals
    function sidewalk()
    {
      ctx.fillStyle = '#343A34';
      ctx.fillRect(0,500,w,10);
      var grd = ctx.createRadialGradient(250, 500, 0,
                                         250, 500, 150);
      grd.addColorStop(0.0, 'rgba(32, 36, 38, .0)');
      grd.addColorStop(0.2, 'rgba(32, 36, 38, 0.1)');
      grd.addColorStop(0.6, 'rgba(32, 36, 38, 0.2)');
      grd.addColorStop(0.8, 'rgba(32, 36, 38, 0.6)');
      grd.addColorStop(1, 'rgba(32, 34, 38, .8)');
      ctx.fillStyle = grd;
      ctx.fillRect(0,500,w,10);
      
      ctx.fillStyle = '#343A34';
      ctx.fillRect(0,510,w,10);
      grd = ctx.createRadialGradient(250, 500, 0,
                                         250, 500, 150);
      grd.addColorStop(0.0, 'rgba(32, 36, 38, 0.7)');
      grd.addColorStop(0.2, 'rgba(32, 36, 38, 0.8)');
      grd.addColorStop(0.4, 'rgba(32, 36, 38, 0.85)');
      grd.addColorStop(1, 'rgba(32, 34, 38, .9)');
      ctx.fillStyle = grd;
      ctx.fillRect(0,510,w,10);
    }
    
    
    // road visuals
    function road() {
      ctx.fillStyle = '#202224';
      ctx.fillRect(0,520,w,h-520);
    }
  
    // function to create a lightning effect on a timer
    function weather(_lTimer) {
  
      lightningAlpha = 0.0;
  
      if ( _lTimer > 350.0) {
        lightningAlpha = (500.0 - _lTimer) * 0.004;
      }
  
      else if (_lTimer < 350.0 && _lTimer > 250.0) {
        lightningAlpha = (_lTimer - 250.0) * 0.006;
      }    
  
      else if (_lTimer < 250.0 && _lTimer >= 100.0) {
        lightningAlpha = (250.0 - _lTimer) * 0.004;
      }
  
      else if (_lTimer < 100.0 && _lTimer >= 0.0) {
        lightningAlpha = _lTimer * 0.006;
      }
  
      if (lightningAlpha > 0.0) {
        ctx.fillStyle = 'rgba(250, 250, 245, ' + lightningAlpha + ')';
        ctx.fillRect(0,0,w,h);
      }
    }
  })