import language_tool_python

tool_en = language_tool_python.LanguageTool('en-US')
tool_de = language_tool_python.LanguageTool('de')

def lan_tool_grammar_check(text, language='en-US'):
    if language == 'en-US':
        return tool_en.check(text)
    if language == 'de':
        return tool_de.check(text)