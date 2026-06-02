//-*-C++-*-
// 4.1.0 버전 수정 파일

#ifndef PDF_STREAM_DECODER_H
#define PDF_STREAM_DECODER_H

namespace pdflib
{

  template<>
  class pdf_decoder<STREAM>
  {

  public:

    pdf_decoder(pdf_resource<PAGE_DIMENSION>& page_dimension_,
                pdf_resource<PAGE_CELLS>&     page_cells_,
                pdf_resource<PAGE_LINES>&     page_lines_,                                   
                pdf_resource<PAGE_IMAGES>&    page_images_,

                pdf_resource<PAGE_FONTS>&     page_fonts_,                                   
                pdf_resource<PAGE_GRPHS>&     page_grphs_,                                   
                pdf_resource<PAGE_XOBJECTS>&  page_xobjects_);

    ~pdf_decoder();

    void print();

    std::set<std::string> get_unknown_operators();

    // decode the qpdf-stream
    void decode(QPDFObjectHandle& content);

    // methods used to interprete the stream
    void interprete(std::vector<qpdf_instruction>& parameters);

  private:

    bool update_stack(std::vector<pdf_state<GLOBAL> >& stack_,
                      int                              stack_count_);

    void interprete(std::vector<qpdf_instruction>& stream_,
                    std::vector<qpdf_instruction>& parameters_);

    void interprete_stream(std::vector<qpdf_instruction>& parameters);

    pdf_state<GLOBAL>& cgs(); // get current global state
    pdf_state<TEXT>&   cts(); // get current text state
    pdf_state<LINE>&   cls(); // get current line state
    pdf_state<GRPH>&   cgrs(); // get current graphics state

    void q();
    void Q();

    void execute_operator(qpdf_instruction op, std::vector<qpdf_instruction> parameters);
    
  private:

    pdf_resource<PAGE_DIMENSION>& page_dimension;
    pdf_resource<PAGE_CELLS>&     page_cells;
    pdf_resource<PAGE_LINES>&     page_lines;
    pdf_resource<PAGE_IMAGES>&    page_images;

    pdf_resource<PAGE_FONTS>&     page_fonts;
    pdf_resource<PAGE_GRPHS>&     page_grphs;
    pdf_resource<PAGE_XOBJECTS>&  page_xobjects;

    std::set<std::string> unknown_operators;

    std::vector<qpdf_instruction> stream;
    std::vector<pdf_state<GLOBAL> > stack;

    int stack_count;
  };

  pdf_decoder<STREAM>::pdf_decoder(pdf_resource<PAGE_DIMENSION>& page_dimension_,
                                   pdf_resource<PAGE_CELLS>&     page_cells_,
                                   pdf_resource<PAGE_LINES>&     page_lines_,                                   
                                   pdf_resource<PAGE_IMAGES>&    page_images_,
                                   
                                   pdf_resource<PAGE_FONTS>&     page_fonts_,
                                   pdf_resource<PAGE_GRPHS>&     page_grphs_,

                                   pdf_resource<PAGE_XOBJECTS>&  page_xobjects_):
    page_dimension(page_dimension_),
    page_cells(page_cells_),    
    page_lines(page_lines_),
    page_images(page_images_),

    page_fonts(page_fonts_),
    page_grphs(page_grphs_),

    page_xobjects(page_xobjects_),

    unknown_operators({}),
    stream({}),
    stack({}),
    
    stack_count(0)
  {
    LOG_S(INFO) << __FUNCTION__;
  }

  pdf_decoder<STREAM>::~pdf_decoder()
  {
    if(unknown_operators.size()>0)
      {
	LOG_S(WARNING) << "============= ~pdf_decoder ===================";
	for(auto item:unknown_operators)
	  {
	    LOG_S(WARNING) << "unknown operator: " << item;
	  }
	LOG_S(WARNING) << "==============================================";
      }
  }

  std::set<std::string> pdf_decoder<STREAM>::get_unknown_operators()
  {
    LOG_S(INFO) << __FUNCTION__;
    return unknown_operators;
  }

  void pdf_decoder<STREAM>::print()
  {
    LOG_S(INFO) << __FUNCTION__;
    for(auto row:stream)
      {
        LOG_S(INFO) << std::setw(12) << row.key << " | " << row.val;
      }
  }

  void pdf_decoder<STREAM>::decode(QPDFObjectHandle& qpdf_content)
  {
    LOG_S(INFO) << __FUNCTION__;

    qpdf_stream_decoder decoder(stream);
    decoder.decode(qpdf_content);
  }

  void pdf_decoder<STREAM>::interprete(std::vector<qpdf_instruction>& parameters)
  {
    LOG_S(INFO) << __FUNCTION__;

    // initialise the stack
    if(stack.size()==0)
      {
        //stack.clear();
        
        pdf_state<GLOBAL> state(page_cells, page_lines, page_images, 
				page_fonts, page_grphs);
        stack.push_back(state);
      }

    interprete_stream(parameters);    
  }

  bool pdf_decoder<STREAM>::update_stack(std::vector<pdf_state<GLOBAL> >& stack_,
                                         int                              stack_count_)
  {
    stack       = stack_;
    stack_count = stack_count_;

    if(stack.size()>0 and page_fonts.keys()!=cgs().page_fonts.keys())
      {
        pdf_state<GLOBAL> state(page_cells, page_lines, page_images, 
				page_fonts, page_grphs);
        state = stack.back();

        stack.push_back(state);              

        return true;
      }

    return false;
  }

  void pdf_decoder<STREAM>::interprete(std::vector<qpdf_instruction>& stream_,
                                       std::vector<qpdf_instruction>& parameters_)
  {
    LOG_S(INFO) << __FUNCTION__;

    stream = stream_;

    interprete_stream(parameters_);

    if(parameters_.size()!=0)
      {
        LOG_S(ERROR) << "Finishing a `Do` with nonzero number of parameters!";
      }
  }

  void pdf_decoder<STREAM>::interprete_stream(std::vector<qpdf_instruction>& parameters)
  {
    LOG_S(INFO) << __FUNCTION__;

    //assert(page_fonts.keys()==cgs().page_fonts.keys());

    for(int l=0; l<stream.size(); l++)
      {
        qpdf_instruction& inst = stream[l];
        
        if(inst.key=="operator")
          {
            // pdf_operator::operator_name  name = pdf_operator::to_name(inst.val); 
            // pdf_operator::operator_class clss = pdf_operator::to_class(name); 
            
	    /*
            if(//clss==pdf_operator::PATH_CONSTRUCTION      or 
               //clss==pdf_operator::PATH_PAINTING          or
               clss==pdf_operator::GENERAL_GRAPHICS_STATE or
               clss==pdf_operator::COLOR_SCHEME            )
              {
                parameters.clear();
                continue;
              }
	    */
	    
	    /*
            for(auto p:parameters)
	      {
		LOG_S(INFO) << "\t" << std::setw(12) << p.key << " | " << p.val;
	      }
            LOG_S(INFO) << " --> " << std::setw(12) << inst.key << " | " << inst.val;
	    */
	    
	    for(auto itr=parameters.begin(); itr!=parameters.end(); )
	      {
		if(itr->key=="null" and itr->val=="null") // this can happen if you have an empty array/dict
		  {
		    LOG_S(ERROR) << "\t" << std::setw(12) << itr->key << " | " << itr->val << " => erasing ...";
		    itr = parameters.erase(itr);
		  }
		else
		  {
		    LOG_S(INFO) << "\t" << std::setw(12) << itr->key << " | " << itr->val;
		    itr++;
		  }
	      }
	    LOG_S(INFO) << " --> " << std::setw(12) << inst.key << " | " << inst.val;
	    
            execute_operator(inst, parameters);

            parameters.clear();
          }
        else
          {
            parameters.push_back(inst);
          }
      }
  }

  // get current global state
  pdf_state<GLOBAL>& pdf_decoder<STREAM>::cgs()
  {
    if(stack.size()==0)
      {
	std::stringstream message;
	message << "stack-size is zero in " << __FILE__ << ":" << __LINE__;

	LOG_S(ERROR) << message.str();
	throw std::logic_error(message.str());
      }

    pdf_state<GLOBAL>& state = stack.back();
    return state;
  }
 
  // get current text state
  pdf_state<TEXT>& pdf_decoder<STREAM>::cts()
  {
    return cgs().text_state;
  } 

  // get current text state
  pdf_state<LINE>& pdf_decoder<STREAM>::cls()
  {
    return cgs().line_state;
  } 

  // get current graphics state
  pdf_state<GRPH>& pdf_decoder<STREAM>::cgrs()
  {
    return cgs().grph_state;
  } 

  /*  
  void pdf_decoder<STREAM>::q()
  {
    pdf_state<GLOBAL> state(page_cells, page_lines, page_images, page_fonts);
    
    if(stack.size()>0)
      {
        state = stack.back();
      }
    
    stack.push_back(state);    

    stack_count += 1;
  } 
  */

  void pdf_decoder<STREAM>::q()
  {
    if(stack.size()==0)
      {
        pdf_state<GLOBAL> state(page_cells, page_lines, page_images, 
				page_fonts, page_grphs);
        stack.push_back(state);      
      }
    else
      {
        pdf_state<GLOBAL> state(stack.back());
        stack.push_back(state);      
      }

    stack_count += 1;
  } 

  void pdf_decoder<STREAM>::Q()
  {
    if(stack.size()>0)
      {
        stack.pop_back();
      }
    else
      {
        LOG_S(ERROR) << "invoking 'Q' on empty stack!";
        //throw std::logic_error(__FILE__);
      }    
  } 

  void pdf_decoder<STREAM>::execute_operator(qpdf_instruction              op, 
                                             std::vector<qpdf_instruction> parameters)
  {
    pdf_operator::operator_name name = pdf_operator::to_name(op.val); 

    switch(name)
      {

        /**************************************************
         ***  General graphics state
         **************************************************/

      case pdf_operator::w:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          cgrs().w(parameters);
        }
        break;

      case pdf_operator::J:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          cgrs().J(parameters);
        }
        break;

      case pdf_operator::j:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          cgrs().j(parameters);
        }
        break;

      case pdf_operator::M:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          cgrs().M(parameters);
        }
        break;

      case pdf_operator::d:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          cgrs().d(parameters);
        }
        break;

      case pdf_operator::ri:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          cgrs().ri(parameters);
        }
        break;

      case pdf_operator::i:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          cgrs().i(parameters);
        }
        break;

      case pdf_operator::gs:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          cgrs().gs(parameters);
        }
        break;

        /**************************************************
         ***  Special graphics state
         **************************************************/

      case pdf_operator::q:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          this->q();
        }
        break;

      case pdf_operator::Q:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          this->Q();
        }
        break;        

      case pdf_operator::cm: 
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cgs().cm(parameters);
        }
        break;

        /**************************************************
         ***  XObjects
         **************************************************/

      case pdf_operator::Do:
        {
          LOG_S(INFO) << "executing " << to_string(name);

          std::string xobj_name = parameters[0].to_utf8_string();

          if(not page_xobjects.has(xobj_name))
            {
              LOG_S(ERROR) << "unknown xobject with name `" << xobj_name << "`";
              return;
            }

          pdf_resource<PAGE_XOBJECT>& xobj = page_xobjects.get(xobj_name);

          switch(xobj.get_subtype())
            {
            case XOBJECT_IMAGE:
              {
                LOG_S(INFO) << "Do_Image: image with `" << xobj_name << "`";
                cgs().Do_image(xobj);
              }
              break;

            case XOBJECT_FORM:
              {
                LOG_S(INFO) << "Do_Form: XObject with name `" << xobj_name << "`";

                pdf_resource<PAGE_FONTS>     page_fonts_ = page_fonts;
                pdf_resource<PAGE_GRPHS>     page_grphs_ = page_grphs;
                pdf_resource<PAGE_XOBJECTS>& page_xobjects_ = page_xobjects;

                // parse the resources of the xobject
                {
                  std::pair<nlohmann::json, QPDFObjectHandle> xobj_fonts = xobj.get_fonts();
                  page_fonts_.set(xobj_fonts.first, xobj_fonts.second);

                  std::pair<nlohmann::json, QPDFObjectHandle> xobj_grphs = xobj.get_grphs();
                  page_grphs_.set(xobj_grphs.first, xobj_grphs.second);

                  std::pair<nlohmann::json, QPDFObjectHandle> xobj_xobjects = xobj.get_xobjects();
                  if(!xobj_xobjects.first.empty()) {
                    bool all_loaded = true;
                    for(auto& pair : xobj_xobjects.first.items()) {
                      if(!page_xobjects.has(pair.key())) { all_loaded = false; break; }
                    }
                    if(!all_loaded) {
                      page_xobjects.set(xobj_xobjects.first, xobj_xobjects.second);
                    }
                  }
                }
                
                {
                  // push-back the stack 
                  this->q();

                  // transform coordinate system
                  cgs().cm(xobj.get_matrix());

                  {
                    std::vector<qpdf_instruction> insts = xobj.parse_stream();

                    pdf_decoder<STREAM> new_stream(page_dimension, page_cells, 
                                                   page_lines, page_images, 
                                                   page_fonts_, page_grphs_, 
						   page_xobjects_);

                    bool updated_stack = new_stream.update_stack(stack, stack_count);

                    // copy the stack
                    std::vector<qpdf_instruction> parameters;
                    new_stream.interprete(insts, parameters);

                    if(updated_stack)
                      {
                        new_stream.Q();
                      }

                    auto unkown_ops = new_stream.get_unknown_operators();
                    for(auto item:unkown_ops)
                      {
                        unknown_operators.insert(item);
                      }
                  }

                  // pop-back the stack 
                  this->Q();                
                }

                LOG_S(INFO) << "ending the execution of FORM XObject with name" << xobj_name;
              }
              break;

            default:
              {
                LOG_S(ERROR) << " unknown subtype of xobject with name " << xobj_name;
              }
            }
        }
        break;

        /**************************************************
         ***  color-schemes
         **************************************************/

      case pdf_operator::CS: 
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().CS(parameters);
        }
        break;

      case pdf_operator::cs: 
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().cs(parameters);
        }
        break;

      case pdf_operator::SC: 
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().SC(parameters);
        }
        break;

      case pdf_operator::SCN: 
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().SCN(parameters);
        }
        break;

      case pdf_operator::sc: 
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().sc(parameters);
        }
        break;

      case pdf_operator::scn: 
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().scn(parameters);
        }
        break;

      case pdf_operator::G: 
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().G(parameters);
        }
        break;

      case pdf_operator::g:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().g(parameters);
        }
        break;

      case pdf_operator::RG: 
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().RG(parameters);
        }
        break;

      case pdf_operator::rg:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().rg(parameters);
        }
        break;
	
      case pdf_operator::K: 
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().K(parameters);
        }
        break;

      case pdf_operator::k:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().k(parameters);
        }
        break;
	
        /**************************************************
         ***  text-objects
         **************************************************/

      case pdf_operator::BT:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          assert(page_fonts.keys()==cgs().page_fonts.keys());

          cts().BT();
        }
        break;

      case pdf_operator::ET:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          cts().ET();
        }
        break;

        /**************************************************
         ***  text-state
         **************************************************/

      case pdf_operator::Tc:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().Tc(parameters);
        }
        break;

      case pdf_operator::Tw:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().Tw(parameters);
        }
        break;

      case pdf_operator::Tz:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().Tz(parameters);
        }
        break;

      case pdf_operator::TL:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().TL(parameters);
        }
        break;

      case pdf_operator::Tf:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().Tf(parameters);
        }
        break;

      case pdf_operator::Tr:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().Tr(parameters);
        }
        break;

      case pdf_operator::Ts:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().Ts(parameters);
        }
        break;

        /**************************************************
         ***  text-positioning
         **************************************************/

      case pdf_operator::Td:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().Td(parameters);
        }
        break;

      case pdf_operator::TD:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().TD(parameters);
        }
        break;

      case pdf_operator::Tm:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().Tm(parameters);
        }
        break;

      case pdf_operator::TStar:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().TStar(parameters);
        }
        break;

        /**************************************************
         ***  text-showing
         **************************************************/

      case pdf_operator::Tj:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().Tj(parameters, stack_count);
        }
        break;

      case pdf_operator::TJ:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().TJ(parameters, stack_count);
        }
        break;

      case pdf_operator::accent:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          assert(parameters.size()==1);

          std::vector<qpdf_instruction> TStar_params = {};
          cts().TStar(TStar_params);
          
          std::vector<qpdf_instruction> Tj_params = {parameters[0]};          
          cts().Tj(Tj_params, stack_count);          
        }
        break;

      case pdf_operator::double_accent:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          assert(parameters.size()==3);

          std::vector<qpdf_instruction> Tw_params = {parameters[0]};
          cts().Tw(Tw_params);
          
          std::vector<qpdf_instruction> Tc_params = {parameters[1]};
          cts().Tc(Tc_params);

          std::vector<qpdf_instruction> TStar_params = {};
          cts().TStar(TStar_params);

          std::vector<qpdf_instruction> Tj_params = {parameters[2]};          
          cts().Tj(Tj_params, stack_count);          
        }
        break;

        /**************************************************
         ***  paths construction [page 132-133]
         **************************************************/

      case pdf_operator::m:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().m(parameters);
        }
        break;

      case pdf_operator::l:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().l(parameters);
        }
        break;

      case pdf_operator::c:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().c(parameters);
        }
        break;

      case pdf_operator::v:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().v(parameters);
        }
        break;

      case pdf_operator::y:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().y(parameters);
        }
        break;

      case pdf_operator::h:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().h(parameters);
        }
        break;

      case pdf_operator::re:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().re(parameters);          
        }
        break;

        /**************************************************
         ***  path painting [page 132-133]
         **************************************************/

      case pdf_operator::s:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().s(parameters);          
        }
        break;

      case pdf_operator::S:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().S(parameters);          
        }
        break;

      case pdf_operator::f:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().f(parameters);          
        }
        break;

      case pdf_operator::F:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().F(parameters);          
        }
        break;

      case pdf_operator::fStar:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().fStar(parameters);          
        }
        break;

      case pdf_operator::B:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().B(parameters);          
        }
        break;

      case pdf_operator::BStar:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().BStar(parameters);          
        }
        break;

      case pdf_operator::b:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().b(parameters);          
        }
        break;

      case pdf_operator::bStar:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().bStar(parameters);          
        }
        break;

      case pdf_operator::n:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().n(parameters);          
        }
        break;

        /**************************************************
         ***  path clipping [page ...]
         **************************************************/

      case pdf_operator::W:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().W(parameters);          
        }
        break;

      case pdf_operator::WStar:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().WStar(parameters);          
        }
        break;

        /**************************************************
         ***  other
         **************************************************/

      case pdf_operator::null:
        {
          LOG_S(WARNING) << "unknown operator with name: " << op.val;
          unknown_operators.insert(op.val);
        }
        break;

      default:
        {
          LOG_S(WARNING) << "ignored operator with name: " << op.val;          
          unknown_operators.insert(op.val);          
        }        
      }
  }

}

#endif
