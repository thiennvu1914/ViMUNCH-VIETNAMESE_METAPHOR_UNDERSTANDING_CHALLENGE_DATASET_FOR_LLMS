from django import template

register = template.Library()

@register.filter(name='replace')
def replace(value, args):
    old, new = args.split(',')
    return value.replace(old, new)
