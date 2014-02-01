using System;
using System.IO;

namespace Clam.NGif
{
    public class LzwEncoder
    {
        private const int Eof = -1;

        private readonly int _imgW;
        private readonly int _imgH;
        private readonly byte[] _pixAry;
        private readonly int _initCodeSize;
        private int _remaining;
        private int _curPixel;

        // GIFCOMPR.C       - GIF Image compression routines
        //
        // Lempel-Ziv compression based on 'compress'.  GIF modifications by
        // David Rowley (mgardi@watdcsu.waterloo.edu)

        // General DEFINEs

        private const int Bits = 12;

        private const int Hsize = 5003; // 80% occupancy

        // GIF Image compression - modified 'compress'
        //
        // Based on: compress.c - File compression ala IEEE Computer, June 1984.
        //
        // By Authors:  Spencer W. Thomas      (decvax!harpo!utah-cs!utah-gr!thomas)
        //              Jim McKie              (decvax!mcvax!jim)
        //              Steve Davies           (decvax!vax135!petsd!peora!srd)
        //              Ken Turkowski          (decvax!decwrl!turtlevax!ken)
        //              James A. Woods         (decvax!ihnp4!ames!jaw)
        //              Joe Orost              (decvax!vax135!petsd!joe)

        int _nBits; // number of bits/code
        private const int Maxbits = Bits; // user settable max # bits/code
        int _maxcode; // maximum code, given n_bits
        private const int Maxmaxcode = 1 << Bits; // should NEVER generate this code

        readonly int[] _htab = new int[Hsize];
        readonly int[] _codetab = new int[Hsize];

        int _freeEnt; // first unused entry

        // block compression parameters -- after all codes are used up,
        // and compression rate changes, start over.
        bool _clearFlg;

        // Algorithm:  use open addressing double hashing (no chaining) on the
        // prefix code / next character combination.  We do a variant of Knuth's
        // algorithm D (vol. 3, sec. 6.4) along with G. Knott's relatively-prime
        // secondary probe.  Here, the modular division first probe is gives way
        // to a faster exclusive-or manipulation.  Also do block compression with
        // an adaptive reset, whereby the code table is cleared when the compression
        // ratio decreases, but after the table fills.  The variable-length output
        // codes are re-sized at this point, and a special CLEAR code is generated
        // for the decompressor.  Late addition:  construct the table according to
        // file size for noticeable speed improvement on small files.  Please direct
        // questions about this implementation to ames!jaw.

        int _gInitBits;

        int _clearCode;
        int _eofCode;

        // output
        //
        // Output the given code.
        // Inputs:
        //      code:   A n_bits-bit integer.  If == -1, then EOF.  This assumes
        //              that n_bits =< wordsize - 1.
        // Outputs:
        //      Outputs code to the file.
        // Assumptions:
        //      Chars are 8 bits long.
        // Algorithm:
        //      Maintain a BITS character long buffer (so that 8 codes will
        // fit in it exactly).  Use the VAX insv instruction to insert each
        // code in turn.  When the buffer fills up empty it and start over.

        int _curAccum;
        int _curBits;

        readonly int[] _masks =
		{
			0x0000,
			0x0001,
			0x0003,
			0x0007,
			0x000F,
			0x001F,
			0x003F,
			0x007F,
			0x00FF,
			0x01FF,
			0x03FF,
			0x07FF,
			0x0FFF,
			0x1FFF,
			0x3FFF,
			0x7FFF,
			0xFFFF };

        // Number of characters so far in this 'packet'
        int _aCount;

        // Define the storage for the packet accumulator
        readonly byte[] _accum = new byte[256];

        //----------------------------------------------------------------------------
        public LzwEncoder(int width, int height, byte[] pixels, int colorDepth)
        {
            _imgW = width;
            _imgH = height;
            _pixAry = pixels;
            _initCodeSize = Math.Max(2, colorDepth);
        }

        // Add a character to the end of the current packet, and if it is 254
        // characters, flush the packet to disk.
        void Add(byte c, Stream outs)
        {
            _accum[_aCount++] = c;
            if (_aCount >= 254)
                Flush(outs);
        }

        // Clear out the hash table

        // table clear for block compress
        void ClearTable(Stream outs)
        {
            ResetCodeTable(Hsize);
            _freeEnt = _clearCode + 2;
            _clearFlg = true;

            Output(_clearCode, outs);
        }

        // reset code table
        void ResetCodeTable(int hsize)
        {
            for (var i = 0; i < hsize; ++i)
                _htab[i] = -1;
        }

        void Compress(int initBits, Stream outs)
        {
            int fcode;
            int c;

            // Set up the globals:  g_init_bits - initial number of bits
            _gInitBits = initBits;

            // Set up the necessary values
            _clearFlg = false;
            _nBits = _gInitBits;
            _maxcode = MaxCode(_nBits);

            _clearCode = 1 << (initBits - 1);
            _eofCode = _clearCode + 1;
            _freeEnt = _clearCode + 2;

            _aCount = 0; // clear packet

            var ent = NextPixel();

            var hshift = 0;
            for (fcode = Hsize; fcode < 65536; fcode *= 2)
                ++hshift;
            hshift = 8 - hshift; // set hash code range bound

            const int hsizeReg = Hsize;
            ResetCodeTable(hsizeReg); // clear hash table

            Output(_clearCode, outs);

        outer_loop: while ((c = NextPixel()) != Eof)
            {
                fcode = (c << Maxbits) + ent;
                var i = (c << hshift) ^ ent /* = 0 */;

                if (_htab[i] == fcode)
                {
                    ent = _codetab[i];
                    continue;
                }
                if (_htab[i] >= 0) // non-empty slot
                {
                    var disp = hsizeReg - i;
                    if (i == 0)
                        disp = 1;
                    do
                    {
                        if ((i -= disp) < 0)
                            i += hsizeReg;

                        if (_htab[i] == fcode)
                        {
                            ent = _codetab[i];
                            goto outer_loop;
                        }
                    } while (_htab[i] >= 0);
                }
                Output(ent, outs);
                ent = c;
                if (_freeEnt < Maxmaxcode)
                {
                    _codetab[i] = _freeEnt++; // code -> hashtable
                    _htab[i] = fcode;
                }
                else
                    ClearTable(outs);
            }
            // Put out the final code.
            Output(ent, outs);
            Output(_eofCode, outs);
        }

        //----------------------------------------------------------------------------
        public void Encode(Stream os)
        {
            os.WriteByte(Convert.ToByte(_initCodeSize)); // write "initial code size" byte

            _remaining = _imgW * _imgH; // reset navigation variables
            _curPixel = 0;

            Compress(_initCodeSize + 1, os); // compress and write the pixel data

            os.WriteByte(0); // write block terminator
        }

        // Flush the packet to disk, and reset the accumulator
        void Flush(Stream outs)
        {
            if (_aCount > 0)
            {
                outs.WriteByte(Convert.ToByte(_aCount));
                outs.Write(_accum, 0, _aCount);
                _aCount = 0;
            }
        }

        static int MaxCode(int nBits)
        {
            return (1 << nBits) - 1;
        }

        //----------------------------------------------------------------------------
        // Return the next pixel from the image
        //----------------------------------------------------------------------------
        private int NextPixel()
        {
            if (_remaining == 0)
                return Eof;

            --_remaining;

            var temp = _curPixel + 1;
            if (temp < _pixAry.GetUpperBound(0))
            {
                var pix = _pixAry[_curPixel++];

                return pix & 0xff;
            }
            return 0xff;
        }

        void Output(int code, Stream outs)
        {
            _curAccum &= _masks[_curBits];

            if (_curBits > 0)
                _curAccum |= (code << _curBits);
            else
                _curAccum = code;

            _curBits += _nBits;

            while (_curBits >= 8)
            {
                Add((byte)(_curAccum & 0xff), outs);
                _curAccum >>= 8;
                _curBits -= 8;
            }

            // If the next entry is going to be too big for the code size,
            // then increase it, if possible.
            if (_freeEnt > _maxcode || _clearFlg)
            {
                if (_clearFlg)
                {
                    _maxcode = MaxCode(_nBits = _gInitBits);
                    _clearFlg = false;
                }
                else
                {
                    ++_nBits;
                    _maxcode = _nBits == Maxbits ? Maxmaxcode : MaxCode(_nBits);
                }
            }

            if (code == _eofCode)
            {
                // At EOF, write the rest of the buffer.
                while (_curBits > 0)
                {
                    Add((byte)(_curAccum & 0xff), outs);
                    _curAccum >>= 8;
                    _curBits -= 8;
                }

                Flush(outs);
            }
        }
    }
}