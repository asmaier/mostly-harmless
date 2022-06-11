function Writer (doc, opts)
	local filter = {
	  Math = function(elem)

		local math = elem

		if elem.mathtype == 'DisplayMath' then
			local delimited = '\n$$' .. elem.text ..'$$\n'
			math = pandoc.RawInline('markdown_mmd', delimited .. '\n')
			-- math = pandoc.RawBlock('markdown', delimited)
		end

		if elem.mathtype == 'InlineMath' then
			local delimited = '$$' .. elem.text ..'$$'
			math = pandoc.RawInline('markdown_mmd', delimited)
		end	
		
		return math
	  end			
	}
	-- print(opts)
	return pandoc.write(doc:walk(filter), 'markdown_mmd', opts)
  end