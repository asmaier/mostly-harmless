function Writer (doc, opts)
	local filter = {
	  Math = function(elem)

		local math = elem

		if elem.mathtype == 'DisplayMath' then
			local delimited = '\n$$' .. elem.text ..'$$\n'
			math = pandoc.RawInline('markdown', delimited .. '\n')
			-- math = pandoc.RawBlock('markdown', delimited)
		end

		if elem.mathtype == 'InlineMath' then
			local delimited = '$$' .. elem.text ..'$$'
			math = pandoc.RawInline('markdown', delimited)
		end	
		
		return math
	  end			
	}
	return pandoc.write(doc:walk(filter), 'markdown', opts)
  end