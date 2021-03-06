#!/usr/bin/env python
#
# Run an application multiple times, varying parameters like
# number of threads, etc

from __future__ import print_function
import sys
import os
import subprocess
import optparse
import shlex
import signal


def die(s):
  sys.stderr.write(s)
  sys.exit(1)


def print_bright(s):
  red = '\033[1;31m'
  endc = '\033[0m'
  print(red + s + endc)


def parse_range(s):
  """
  Parses thread range s
  Grammar:
   R := R,R
      | S
      | N
      | N:N
      | N:N:N
   N := an integer
   S := a string
  """
  # Parsing strategy: greedily parse integers with one character
  # lookahead to figure out exact category
  s = s + ' ' # append special end marker
  retval = []
  cur = -1
  curseq = []
  for i in range(len(s)):
    if s[i] == ',' or s[i] == ' ':
      if cur < 0:
        break
      if len(curseq) == 0:
        retval.append(s[cur:i])
      elif len(curseq) == 1:
        retval.extend(range(curseq[0], int(s[cur:i]) + 1))
      elif len(curseq) == 2:
        retval.extend(range(curseq[0], curseq[1] + 1, int(s[cur:i])))
      else:
        break
      cur = -1
      curseq = []
    elif s[i] == ':' and cur >= 0:
      curseq.append(int(s[cur:i]))
      cur = -1
    elif cur < 0:
      cur = i
    else:
      pass
  else:
    return sorted(set(retval))
  die('error parsing range: %s\n' % s)


def product(args):
  """
  Like itertools.product but for one iterable of iterables
  rather than an argument list of iterables
  """
  pools = map(tuple, args)
  result = [[]]
  for pool in pools:
    result = [x+[y] for x in result for y in pool]
  for prod in result:
    yield tuple(prod)


def run(cmd, values, options):
  import subprocess, datetime, os, time, signal, socket

  for R in range(options.runs):
    print('RUN: Start')
    print_bright('RUN: Start')
    print("RUN: CommandLine %s" % ' '.join(cmd))
    print("RUN: Variable Hostname = %s" % socket.gethostname())
    print("RUN: Variable Timestamp = %f" % time.time())

    for (name, value) in values:
      print('RUN: Variable %s = %s' % (name, value))

    if options.timeout:
      start = datetime.datetime.now()
      process = subprocess.Popen(cmd)
      while process.poll() is None:
        time.sleep(5)
        now = datetime.datetime.now()
        diff = (now-start).seconds
        if diff > options.timeout:
          os.kill(process.pid, signal.SIGKILL)
          #os.waitpid(-1, os.WNOHANG)
          os.waitpid(-1, 0)
          print("RUN: Variable Timeout = %d" % (diff*1000))
          break
      retcode = process.returncode
    else:
      retcode = subprocess.call(cmd)
    if retcode != 0:
      # print command line just in case child process should be died before doing it
      print("RUN: Error %s" % retcode)
      if not options.ignore_errors:
        sys.exit(1)

def parse_extra(extra):
  """
  Parse extra command line option.
  
  Three cases:
   (1) <name>::<arg>::<range>
   (2) ::<arg>::<range>
   (3) <name>::<range>
  """
  import re
  if extra.count('::') == 2:
    (name, arg, r) = extra.split('::')
    if not name:
      name = re.sub(r'^-*', '', arg)
  elif extra.count('::') == 1:
    (name, r) = extra.split('::')
    arg = None
  else:
    die('error parsing extra argument: %s\n' % extra)
  return (name, arg, r)


def main(args, options):
  variables = []
  ranges = []
  for extra in options.extra:
    (name, arg, r) = parse_extra(extra)
    variables.append((name, arg))
    ranges.append(parse_range(r))

  if not ranges and options.no_default_thread:
    run(args, [], options)
  else:
    for prod in product(ranges):
      cmd = [args[0]]
      values = []
      for ((name, arg), value) in zip(variables, prod):
        if arg:
          cmd.extend([arg, str(value)])
        else:
          cmd.extend([str(value)])
        values.append((name, str(value)))
      cmd.extend(args[1:])

      run(cmd, values, options)


if __name__ == '__main__':
  signal.signal(signal.SIGQUIT, signal.SIG_IGN)
  sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
  parser = optparse.OptionParser(usage='usage: %prog [options] <command line> ...')
  parser.add_option('--ignore-errors', dest='ignore_errors', default=False, action='store_true',
      help='ignore errors in subprocesses')
  parser.add_option('-t', '--threads', dest="threads", default="1",
      help='range of threads to use. A range is R := R,R | S | N | N:N | N:N:N where N is an integer and S is a string.')
  parser.add_option('-r', '--runs', default=1, type='int',
      help='set number of runs')
  parser.add_option('-x', '--extra', dest="extra", default=[], action='append',
      help='add another parameter to range over (format: <name>::<arg>::<range> or ::<arg>::<range> or <name>::<range>). E.g., delta::-delta::1,5 or ::-delta::1,5 or schedule::-useFIFO,-useLIFO')
  parser.add_option('-o', '--timeout', dest="timeout", default=0, type='int',
      help="timeout a run after SEC seconds", metavar='SEC')
  parser.add_option('--no-default-thread', dest='no_default_thread', default=False, action='store_true',
      help='run command default thread parameter')
  (options, args) = parser.parse_args()
  if not args:
    parser.error('need command to run')
  if not options.no_default_thread:
    options.extra.insert(0, '%s::%s::%s' % ('Threads', '-t', options.threads))
  main(args, options)
