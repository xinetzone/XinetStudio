#-----------------------------------------------------------------------
# useargument.py
#-----------------------------------------------------------------------
from booksite import stdio
import sys

# Accept a name as a command-line argument. Write a message containing
# that name to standard output.

stdio.write('Hi, ')    # 不换行
stdio.write(sys.argv[1])
stdio.writeln('. How are you?')  # 换行

#-----------------------------------------------------------------------

# python useargument.py Alice
# Hi, Alice. How are you?

# python useargument.py Bob
# Hi, Bob. How are you?

# python useargument.py Carol
# Hi, Carol. How are you?
