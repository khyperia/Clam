namespace Clam.NGif
{
	public class NeuQuant 
	{
	    private const int Netsize = 256; /* number of colours used */
	    /* four primes near 500 - assume no image has a length so large */
		/* that it is divisible by all four primes */
	    private const int Prime1 = 499;
	    private const int Prime2 = 491;
	    private const int Prime3 = 487;
	    private const int Prime4 = 503;
	    private const int Minpicturebytes = (3*Prime4);
	    /* minimum size for input image */
		/* Program Skeleton
		   ----------------
		   [select samplefac in range 1..30]
		   [read image from input file]
		   pic = (unsigned char*) malloc(3*width*height);
		   initnet(pic,3*width*height,samplefac);
		   learn();
		   unbiasnet();
		   [write output image header, using writecolourmap(f)]
		   inxbuild();
		   write output image using inxsearch(b,g,r)      */

		/* Network Definitions
		   ------------------- */
	    private const int Maxnetpos = (Netsize - 1);
	    private const int Netbiasshift = 4; /* bias for colour values */
	    private const int Ncycles = 100; /* no. of learning cycles */

	    /* defs for freq and bias */
	    private const int Intbiasshift = 16; /* bias for fractions */
	    private const int Intbias = (1 << Intbiasshift);
	    private const int Gammashift = 10; /* gamma = 1024 */
	    private const int Betashift = 10;
	    private const int Beta = (Intbias >> Betashift); /* beta = 1/1024 */

	    private const int Betagamma = (Intbias << (Gammashift - Betashift));

	    /* defs for decreasing radius factor */
	    private const int Initrad = (Netsize >> 3); /* for 256 cols, radius starts */
	    private const int Radiusbiasshift = 6; /* at 32.0 biased by 6 bits */
	    private const int Radiusbias = (1 << Radiusbiasshift);
	    private const int Initradius = (Initrad*Radiusbias); /* and decreases by a */
	    private const int Radiusdec = 30; /* factor of 1/30 each cycle */

	    /* defs for decreasing alpha factor */
	    private const int Alphabiasshift = 10; /* alpha starts at 1.0 */
	    private const int Initalpha = (1 << Alphabiasshift);

	    private int _alphadec; /* biased by 10 bits */

		/* radbias and alpharadbias used for radpower calculation */
	    private const int Radbiasshift = 8;
	    private const int Radbias = (1 << Radbiasshift);
	    private const int Alpharadbshift = (Alphabiasshift + Radbiasshift);
	    private const int Alpharadbias = (1 << Alpharadbshift);

	    /* Types and Global Variables
		-------------------------- */

	    private readonly byte[] _thepicture; /* the input image itself */
	    private readonly int _lengthcount; /* lengthcount = H*W*3 */

	    private int _samplefac; /* sampling factor 1..30 */

		//   typedef int pixel[4];                /* BGRc */
	    private readonly int[][] _network; /* the network itself - [netsize][4] */

	    private readonly int[] _netindex = new int[256];
		/* for network lookup - really 256 */

	    private readonly int[] _bias = new int[Netsize];
		/* bias and freq arrays for learning */
	    private readonly int[] _freq = new int[Netsize];
	    private readonly int[] _radpower = new int[Initrad];
		/* radpower for precomputation */

		/* Initialise network in range (0,0,0) to (255,255,255) and set parameters
		   ----------------------------------------------------------------------- */
		public NeuQuant(byte[] thepic, int len, int sample) 
		{

			int i;

		    _thepicture = thepic;
			_lengthcount = len;
			_samplefac = sample;

			_network = new int[Netsize][];
			for (i = 0; i < Netsize; i++) 
			{
				_network[i] = new int[4];
				var p = _network[i];
				p[0] = p[1] = p[2] = (i << (Netbiasshift + 8)) / Netsize;
				_freq[i] = Intbias / Netsize; /* 1/netsize */
				_bias[i] = 0;
			}
		}

	    private byte[] ColorMap() 
		{
			var map = new byte[3 * Netsize];
			var index = new int[Netsize];
			for (var i = 0; i < Netsize; i++)
				index[_network[i][3]] = i;
			var k = 0;
			for (var i = 0; i < Netsize; i++) 
			{
				var j = index[i];
				map[k++] = (byte) (_network[j][0]);
				map[k++] = (byte) (_network[j][1]);
				map[k++] = (byte) (_network[j][2]);
			}
			return map;
		}
	
		/* Insertion sort of network and building of netindex[0..255] (to do after unbias)
		   ------------------------------------------------------------------------------- */

	    private void Inxbuild() 
		{
	        int i, j;

	        var previouscol = 0;
			var startpos = 0;
			for (i = 0; i < Netsize; i++) 
			{
				var p = _network[i];
				var smallpos = i;
				var smallval = p[1];
				/* find smallest in i..netsize-1 */
			    int[] q;
			    for (j = i + 1; j < Netsize; j++) 
				{
					q = _network[j];
					if (q[1] < smallval) 
					{ /* index on g */
						smallpos = j;
						smallval = q[1]; /* index on g */
					}
				}
				q = _network[smallpos];
				/* swap p (i) and q (smallpos) entries */
				if (i != smallpos) 
				{
					j = q[0];
					q[0] = p[0];
					p[0] = j;
					j = q[1];
					q[1] = p[1];
					p[1] = j;
					j = q[2];
					q[2] = p[2];
					p[2] = j;
					j = q[3];
					q[3] = p[3];
					p[3] = j;
				}
				/* smallval entry is now in position i */
				if (smallval != previouscol) 
				{
					_netindex[previouscol] = (startpos + i) >> 1;
					for (j = previouscol + 1; j < smallval; j++)
						_netindex[j] = i;
					previouscol = smallval;
					startpos = i;
				}
			}
			_netindex[previouscol] = (startpos + Maxnetpos) >> 1;
			for (j = previouscol + 1; j < 256; j++)
				_netindex[j] = Maxnetpos; /* really 256 */
		}
	
		/* Main Learning Loop
		   ------------------ */

	    private void Learn() 
		{
	        int i;
	        int step;

	        if (_lengthcount < Minpicturebytes)
				_samplefac = 1;
			_alphadec = 30 + ((_samplefac - 1) / 3);
			var p = _thepicture;
			var pix = 0;
			var lim = _lengthcount;
			var samplepixels = _lengthcount / (3 * _samplefac);
			var delta = samplepixels / Ncycles;
			var alpha = Initalpha;
			var radius = Initradius;

			var rad = radius >> Radiusbiasshift;
			if (rad <= 1)
				rad = 0;
			for (i = 0; i < rad; i++)
				_radpower[i] =
					alpha * (((rad * rad - i * i) * Radbias) / (rad * rad));

			//fprintf(stderr,"beginning 1D learning: initial radius=%d\n", rad);

			if (_lengthcount < Minpicturebytes)
				step = 3;
			else if ((_lengthcount % Prime1) != 0)
				step = 3 * Prime1;
			else 
			{
				if ((_lengthcount % Prime2) != 0)
					step = 3 * Prime2;
				else 
				{
					if ((_lengthcount % Prime3) != 0)
						step = 3 * Prime3;
					else
						step = 3 * Prime4;
				}
			}

			i = 0;
			while (i < samplepixels) 
			{
				var b = (p[pix + 0] & 0xff) << Netbiasshift;
				var g = (p[pix + 1] & 0xff) << Netbiasshift;
				var r = (p[pix + 2] & 0xff) << Netbiasshift;
				var j = Contest(b, g, r);

				Altersingle(alpha, j, b, g, r);
				if (rad != 0)
					Alterneigh(rad, j, b, g, r); /* alter neighbours */

				pix += step;
				if (pix >= lim)
					pix -= _lengthcount;

				i++;
				if (delta == 0)
					delta = 1;
				if (i % delta == 0) 
				{
					alpha -= alpha / _alphadec;
					radius -= radius / Radiusdec;
					rad = radius >> Radiusbiasshift;
					if (rad <= 1)
						rad = 0;
					for (j = 0; j < rad; j++)
						_radpower[j] =
							alpha * (((rad * rad - j * j) * Radbias) / (rad * rad));
				}
			}
			//fprintf(stderr,"finished 1D learning: readonly alpha=%f !\n",((float)alpha)/initalpha);
		}
	
		/* Search for BGR values 0..255 (after net is unbiased) and return colour index
		   ---------------------------------------------------------------------------- */
		public int Map(int b, int g, int r) 
		{
		    var bestd = 1000;
			var best = -1;
			var i = _netindex[g];
			var j = i - 1;

			while ((i < Netsize) || (j >= 0)) 
			{
			    int dist;
			    int a;
			    int[] p;
			    if (i < Netsize) 
				{
					p = _network[i];
					dist = p[1] - g; /* inx key */
					if (dist >= bestd)
						i = Netsize; /* stop iter */
					else 
					{
						i++;
						if (dist < 0)
							dist = -dist;
						a = p[0] - b;
						if (a < 0)
							a = -a;
						dist += a;
						if (dist < bestd) 
						{
							a = p[2] - r;
							if (a < 0)
								a = -a;
							dist += a;
							if (dist < bestd) 
							{
								bestd = dist;
								best = p[3];
							}
						}
					}
				}
				if (j >= 0) 
				{
					p = _network[j];
					dist = g - p[1]; /* inx key - reverse dif */
					if (dist >= bestd)
						j = -1; /* stop iter */
					else 
					{
						j--;
						if (dist < 0)
							dist = -dist;
						a = p[0] - b;
						if (a < 0)
							a = -a;
						dist += a;
						if (dist < bestd) 
						{
							a = p[2] - r;
							if (a < 0)
								a = -a;
							dist += a;
							if (dist < bestd) 
							{
								bestd = dist;
								best = p[3];
							}
						}
					}
				}
			}
			return (best);
		}
		public byte[] Process() 
		{
			Learn();
			Unbiasnet();
			Inxbuild();
			return ColorMap();
		}
	
		/* Unbias network to give byte values 0..255 and record position i to prepare for sort
		   ----------------------------------------------------------------------------------- */

	    private void Unbiasnet() 
		{

			int i;

			for (i = 0; i < Netsize; i++) 
			{
				_network[i][0] >>= Netbiasshift;
				_network[i][1] >>= Netbiasshift;
				_network[i][2] >>= Netbiasshift;
				_network[i][3] = i; /* record colour no */
			}
		}
	
		/* Move adjacent neurons by precomputed alpha*(1-((i-j)^2/[r]^2)) in radpower[|i-j|]
		   --------------------------------------------------------------------------------- */

	    private void Alterneigh(int rad, int i, int b, int g, int r) 
		{
	        var lo = i - rad;
			if (lo < -1)
				lo = -1;
			var hi = i + rad;
			if (hi > Netsize)
				hi = Netsize;

			var j = i + 1;
			var k = i - 1;
			var m = 1;
			while ((j < hi) || (k > lo)) 
			{
				var a = _radpower[m++];
			    int[] p;
			    if (j < hi) 
				{
					p = _network[j++];
					try 
					{
						p[0] -= (a * (p[0] - b)) / Alpharadbias;
						p[1] -= (a * (p[1] - g)) / Alpharadbias;
						p[2] -= (a * (p[2] - r)) / Alpharadbias;
					} 
					catch
					{
					} // prevents 1.3 miscompilation
				}
				if (k > lo) 
				{
					p = _network[k--];
					try 
					{
						p[0] -= (a * (p[0] - b)) / Alpharadbias;
						p[1] -= (a * (p[1] - g)) / Alpharadbias;
						p[2] -= (a * (p[2] - r)) / Alpharadbias;
					} 
					catch
					{
					}
				}
			}
		}
	
		/* Move neuron i towards biased (b,g,r) by factor alpha
		   ---------------------------------------------------- */

	    private void Altersingle(int alpha, int i, int b, int g, int r) 
		{

			/* alter hit neuron */
			var n = _network[i];
			n[0] -= (alpha * (n[0] - b)) / Initalpha;
			n[1] -= (alpha * (n[1] - g)) / Initalpha;
			n[2] -= (alpha * (n[2] - r)) / Initalpha;
		}
	
		/* Search for biased BGR values
		   ---------------------------- */

	    private int Contest(int b, int g, int r) 
		{

			/* finds closest neuron (min dist) and updates freq */
			/* finds best neuron (min dist-bias) and returns position */
			/* for frequently chosen neurons, freq[i] is high and bias[i] is negative */
			/* bias[i] = gamma*((1/netsize)-freq[i]) */

	        int i;

	        var bestd = ~(1 << 31);
			var bestbiasd = bestd;
			var bestpos = -1;
			var bestbiaspos = bestpos;

			for (i = 0; i < Netsize; i++) 
			{
				var n = _network[i];
				var dist = n[0] - b;
				if (dist < 0)
					dist = -dist;
				var a = n[1] - g;
				if (a < 0)
					a = -a;
				dist += a;
				a = n[2] - r;
				if (a < 0)
					a = -a;
				dist += a;
				if (dist < bestd) 
				{
					bestd = dist;
					bestpos = i;
				}
				var biasdist = dist - ((_bias[i]) >> (Intbiasshift - Netbiasshift));
				if (biasdist < bestbiasd) 
				{
					bestbiasd = biasdist;
					bestbiaspos = i;
				}
				var betafreq = (_freq[i] >> Betashift);
				_freq[i] -= betafreq;
				_bias[i] += (betafreq << Gammashift);
			}
			_freq[bestpos] += Beta;
			_bias[bestpos] -= Betagamma;
			return (bestbiaspos);
		}
	}
}