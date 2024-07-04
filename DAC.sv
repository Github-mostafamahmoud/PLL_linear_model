module DAC #(parameter DATA_WIDTH = 18 )
  (
    input logic reset,
    input signed [DATA_WIDTH-1:0] data,
    output real analog_outp,
    output real analog_outn,
    output real max_analog_outp,
    output real max_analog_outn
  );

  // Internal signal for holding the analog output
   reg signed [DATA_WIDTH-1:0] dac_positive_output ;
   reg signed [DATA_WIDTH-1:0] dac_negative_output ;
   real RES = (1.2/2**DATA_WIDTH);//assume Vref=1.2 Binary wghited DAC
   reg [DATA_WIDTH-1:0] data1;
  // DAC implementation
  always_ff @(data, posedge reset) begin
    if (reset) begin
      dac_positive_output <= 0;
      dac_negative_output <= 0;
    end else begin
	   data1 = data + 2**(DATA_WIDTH-1);
	   data1 = data1>>1 ;
           dac_positive_output <= data1 ;
           dac_negative_output <= data1  ;

    end
  end

  // Assign analog output
  assign analog_outp = dac_positive_output*(1.2/2**(DATA_WIDTH)); // assume vref=1.2
  assign analog_outn = -dac_negative_output*(1.2/2**(DATA_WIDTH)); // assume vref=1.2
  assign max_analog_outp = (131071)**(1.2/2**(DATA_WIDTH));
  assign max_analog_outn = -(131071)**(1.2/2**(DATA_WIDTH));
endmodule 
