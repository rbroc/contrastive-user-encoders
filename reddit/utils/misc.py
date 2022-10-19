def stringify(lst):
    ''' List to string for SQL query '''
    return '(' + ', '.join([f'\'{l}\'' for l in lst]) + ')'
