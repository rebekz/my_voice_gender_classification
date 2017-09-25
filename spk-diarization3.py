#!/usr/bin/env python2
import argparse
import sys
from os import getcwd
import os.path as op
from tempfile import mkstemp, gettempdir
from subprocess import Popen, call
from mimetypes import guess_type
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a media file to perform\
                                     segmentation and speaker clustering on it.')
    parser.add_argument('infile', type=str,
                        help='Specifies the media file')
    parser.add_argument('-o', dest='outfile', type=str, default='stdout',
                        help='Specifies an output recipe file, default stdout.')
    # parser.add_argument('-of', dest='outformat', type=str,
    #                     choices=['aku', 'elan', 'ann'], default='aku',
    #                     help='Specifies an output format, defaults to aku recipe\
    #                     file, but includes support for ELAN .eaf and simple\
    #                     annotation files.')
    parser.add_argument('-fc', dest='fcpath', type=str, default=getcwd(),
                        help='Specifies the path to feacat, defaults to ./')
    parser.add_argument('-fcfg', dest='fcfg', type=str, default=getcwd() + '/fconfig.cfg',
                        help='Specifies the feacat acoustic model config, defaults ./fconfig.cfg')
    parser.add_argument('-lna', dest='lnapath', type=str, default=getcwd() + '/lna',
                        help='Specifies the path to the lna files, defaults to ./lna')
    parser.add_argument('-exp', dest='exppath', type=str, default=getcwd() + '/exp',
                        help='Specifies the path to the exp files, defaults to ./exp')
    parser.add_argument('-fp', dest='feapath', type=str, default=getcwd() + '/fea',
                        help='Specifies the path to the feature files, defaults to ./fea')
    parser.add_argument('-tmp', dest='tmppath', type=str, default='',
                        help='Specifies where to write the temporal files, defaults to system temporary folder.')
    args = parser.parse_args()

    # Process arguments
    if not op.isfile(args.infile):
        print '%s does not exist, exiting' % args.infile
        sys.exit()
    print 'Reading file:', args.infile

    if args.outfile != 'stdout':
        outfile = args.outfile
        print 'Writing output to:', args.outfile
    else:
        outfile = 'stdout'
        print 'Writing output to: stdout'

    args.fcpath = op.join(args.fcpath, 'feacat')
    if not op.isfile(args.fcpath):
        print '%s does not exist, exiting' % args.fcpath
        sys.exit()
    print 'Using feacat from:', args.fcpath

    if not op.isdir(args.tmppath):
        args.tmppath = gettempdir()
    print 'Writing temporal files in:', args.tmppath

    if not op.isdir(args.lnapath):
        print 'Path %s does not exist, exiting' % args.lnapath
        sys.exit()
    print 'Writing lna files in:', args.lnapath

    if not op.isdir(args.exppath):
        print 'Path %s does not exist, exiting' % args.exppath
        sys.exit()
    print 'Writing exp files in:', args.exppath

    if not op.isdir(args.exppath):
        print 'Path %s does not exist, exiting' % args.feapath
        sys.exit()
    print 'Writing features in:', args.feapath
    # End of argument processing

    # Checking if media file is .wav audio
    mediatype = guess_type(args.infile)[0]
    if mediatype != 'audio/x-wav':
        print 'Media is not a .wav audio file, attempting to extract a .wav file'
        print 'Calling ffmpeg'
        infile = op.splitext(args.infile)[0] + '.wav'
        call(['ffmpeg', '-i', args.infile, '-ar', '16000', '-ac', '1',
             '-ab', '32k', infile])
    else:
        infile = args.infile

    # Prepare an initial temporal recipe
    init_recipe = mkstemp(suffix='.recipe', prefix='init', dir=args.tmppath)[1]
    init_file = open(init_recipe, 'w')
    init_file.write('audio=' + infile + '\n')
    init_file.close()

    print 'Performing exp generation and feacat concurrently'
    child1 = Popen(['./generate_exp.py', init_recipe, '-e', args.exppath,
                    '-l', args.lnapath])

    feafile = open(op.join(args.feapath, op.splitext(op.basename(infile))[0] + '.fea'), 'w')
    child2 = Popen([args.fcpath, '-c', args.fcfg, '-H', '--raw-output',
                    infile], stdout=feafile)

    # We need the exp files ready here
    child1.wait()

    def _formatf(f):
        return str(f).replace('.', '')

    print 'Calling voice-detection2.py'
    mediafile = op.splitext(op.basename(args.infile))[0]
    vad_recipe = mkstemp(suffix='.recipe', prefix='vad', dir=args.tmppath)[1]
    call(['./voice-detection2.py', init_recipe, args.exppath,
        '-o', vad_recipe, '-ms', '0.1', '-mns', '1.5'])

    # We need to wait for the features to be ready here
    print 'Waiting for feacat to end.'
    child2.wait()
    # sys.exit()

    spkchange_recipe = mkstemp(suffix='.recipe', prefix='spkc',
                               dir=args.tmppath)[1]
    print 'Calling spk-change-detection.py'
    call(['./spk-change-detection.py', vad_recipe, args.feapath,
        '-o', spkchange_recipe, '-m', 'gw', '-d', 'BIC', '-w', '1.0',
        '-st', '3.0', '-dws', '0.1', '-l', '0.75'])

    print 'Calling spk-clustering.py'
    call(['./spk-clustering.py', spkchange_recipe, args.feapath,
          '-o', outfile, '-m', 'hi', '-d', 'GLR', '-t', '5000'])

    # Outputting alternative formats
    if outfile != 'stdout':
        outf = op.splitext(op.basename(outfile))[0]
        outfpath = op.dirname(outfile)
        print 'Calling aku2ann.py'
        call(['./aku2ann.py', outfile, '-o', op.join(outfpath, outf + '.ann')])

        print 'Calling aku2elan.py'
        call(['./aku2elan.py', outfile, '-o', op.join(outfpath, outf + '.eaf')])
