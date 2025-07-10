"""
Test for OMML fraction processing with missing fPr elements
"""

import lxml.etree as ET
from docling.backend.docx.latex.omml import oMath2Latex, OMML_NS


def test_omml_fraction_missing_fpr():
    """Test that fractions with missing fPr elements are handled gracefully"""
    
    # Create an OMML fraction without fPr element (this would cause KeyError before fix)
    omml_xml = f'''
    <math:oMath xmlns:math="{OMML_NS[1:-1]}">
        <math:f>
            <math:num>
                <math:r>
                    <math:t>a</math:t>
                </math:r>
            </math:num>
            <math:den>
                <math:r>
                    <math:t>b</math:t>
                </math:r>
            </math:den>
        </math:f>
    </math:oMath>
    '''
    
    # Parse and process the OMML - this should not raise KeyError
    root = ET.fromstring(omml_xml)
    result = oMath2Latex(root)
    latex_result = str(result)
    
    # Should produce default fraction formatting
    assert "\\frac{a}{b}" in latex_result


def test_omml_fraction_with_fpr():
    """Test that fractions with fPr elements still work correctly"""
    
    # Create an OMML fraction with fPr element
    omml_xml = f'''
    <math:oMath xmlns:math="{OMML_NS[1:-1]}">
        <math:f>
            <math:fPr>
                <math:type math:val="bar"/>
            </math:fPr>
            <math:num>
                <math:r>
                    <math:t>x</math:t>
                </math:r>
            </math:num>
            <math:den>
                <math:r>
                    <math:t>y</math:t>
                </math:r>
            </math:den>
        </math:f>
    </math:oMath>
    '''
    
    # Parse and process the OMML
    root = ET.fromstring(omml_xml)
    result = oMath2Latex(root)
    latex_result = str(result)
    
    # Should produce fraction formatting with fPr properties
    assert "\\frac{x}{y}" in latex_result


def test_omml_fraction_missing_components():
    """Test that fractions with missing numerator or denominator use placeholders"""
    
    # Create an OMML fraction with missing numerator
    omml_xml_missing_num = f'''
    <math:oMath xmlns:math="{OMML_NS[1:-1]}">
        <math:f>
            <math:den>
                <math:r>
                    <math:t>b</math:t>
                </math:r>
            </math:den>
        </math:f>
    </math:oMath>
    '''
    
    # Parse and process the OMML
    root = ET.fromstring(omml_xml_missing_num)
    result = oMath2Latex(root)
    latex_result = str(result)
    
    # Should use placeholder for missing numerator
    assert "formula_skipped" in latex_result
    assert "b" in latex_result


def test_omml_complex_equation_with_missing_fpr():
    """Test that complex equations with missing fPr continue processing"""
    
    # Create a complex equation with a fraction missing fPr
    omml_xml = f'''
    <math:oMath xmlns:math="{OMML_NS[1:-1]}">
        <math:r>
            <math:t>x = </math:t>
        </math:r>
        <math:f>
            <math:num>
                <math:r>
                    <math:t>a + b</math:t>
                </math:r>
            </math:num>
            <math:den>
                <math:r>
                    <math:t>c</math:t>
                </math:r>
            </math:den>
        </math:f>
        <math:r>
            <math:t> + y</math:t>
        </math:r>
    </math:oMath>
    '''
    
    # Parse and process the OMML
    root = ET.fromstring(omml_xml)
    result = oMath2Latex(root)
    latex_result = str(result)
    
    # Should process the entire equation correctly
    assert "x =" in latex_result
    assert "\\frac{a + b}{c}" in latex_result
    assert "+ y" in latex_result


if __name__ == "__main__":
    import logging
    
    # Set up logging to capture warnings
    logging.basicConfig(level=logging.WARNING)
    
    # Run all tests
    test_omml_fraction_missing_fpr()
    test_omml_fraction_with_fpr()
    test_omml_fraction_missing_components()
    test_omml_complex_equation_with_missing_fpr()
    
    print("All OMML fraction tests passed!")