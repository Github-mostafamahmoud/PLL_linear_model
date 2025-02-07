/*module TDC #(parameter real RESOLUTION = 1e-13,parameter integer DOUT_WIDTH = 7 , parameter real time_unit)(
  input wire Clk_ref,
  input wire feedback,
  output reg signed [DOUT_WIDTH-1:0] Dout
);
real time_diff_neg,time_diff_neg_degital;
real Tfs = RESOLUTION*((2**(DOUT_WIDTH-1))-1);
  real time_ref_P,time_ref_N, time_fb_P,time_fb_N, time_diff;
  logic clk ;
  int count_ref, count_fb; 
  
  function automatic real abs_val(input real num);
  if (num < 0) begin
      return -num;
  end else begin
      return num;
  end
  endfunction
function int round_to_nearest(real x);
    return int'(x); // implicit conversion
  endfunction
 /* function automatic integer round_to_nearest(input real num);
  if (num < 0) begin
      return $rtoi($ceil(num - 0.5));
  end else begin
      return $rtoi($floor(num + 0.5));
  end
  endfunction
*/ 
//ONE TO REMOVE
/* 
  always_ff @(posedge Clk_ref) begin
      time_ref_P <= $realtime;
      count_ref <= 1;
      if (count_ref&count_fb)begin
      count_ref=0;
      count_fb=0;
    end
  end
 
 
  always_ff @(posedge feedback) begin 
      time_fb_P <= $realtime;
      count_fb <= 1;
      if (count_ref&count_fb)begin
      count_ref=0;
      count_fb=0;
     end
      
  end
 

  always_comb begin
  if ((count_ref == 1) && (count_fb == 1)) begin
      time_diff = (time_fb_P - time_ref_P) * time_unit;
      if(abs_val(time_diff) <= Tfs) begin
          Dout = round_to_nearest(time_diff / RESOLUTION);  
      end
      else if (time_diff > 0) begin
        
          Dout = round_to_nearest(Tfs / RESOLUTION);
          
      end
      else begin
        
         Dout = round_to_nearest((-Tfs / RESOLUTION)-1);
          
      end
  end
  time_diff_neg=-1.0*time_diff;
  time_diff_neg_degital = time_diff /RESOLUTION;
end

endmodule
*/

`timescale 1ps/1fs
module TDC #(parameter real RESOLUTION = 1e-13,parameter integer DOUT_WIDTH = 7)(
  input wire Clk_ref,
  input wire feedback,
  output reg signed [DOUT_WIDTH-1:0] Dout,
  output real time_diff
);
real time_diff_neg,time_diff_neg_degital;
real Tfs = RESOLUTION*((2**(DOUT_WIDTH-1))-1);
  real time_ref_P,time_ref_N, time_fb_P,time_fb_N,time_unit; //time_diff
  logic clk ;
  int count_ref, count_fb; 
  assign time_unit = 1e-12; 
  function automatic real abs_val(input real num);
  if (num < 0) begin
      return -num;
  end else begin
      return num;
  end
  endfunction
function int round_to_nearest(real x);
    return int'(x); // implicit conversion
  endfunction
 /* function automatic integer round_to_nearest(input real num);
  if (num < 0) begin
      return $rtoi($ceil(num - 0.5));
  end else begin
      return $rtoi($floor(num + 0.5));
  end
  endfunction
*/  
  always_ff @(posedge Clk_ref) begin
      time_ref_P <= $realtime;
      count_ref <= 1;
      if (count_ref&count_fb)begin
      count_ref=0;
      count_fb=0;
    end
  end
 
 
  always_ff @(posedge feedback) begin 
      time_fb_P <= $realtime;
      count_fb <= 1;
      if (count_ref&count_fb)begin
      count_ref=0;
      count_fb=0;
     end
      
  end
 

  always_comb begin
  if ((count_ref == 1) && (count_fb == 1)) begin
      time_diff = (time_fb_P - time_ref_P) * time_unit;
      if(abs_val(time_diff) <= Tfs) begin
          Dout = round_to_nearest(time_diff / RESOLUTION);  
      end
      else if (time_diff > 0) begin
        
          Dout = round_to_nearest(Tfs / RESOLUTION);
          
      end
      else begin
        
         Dout = round_to_nearest((-Tfs / RESOLUTION)-1);
          
      end
  end
  time_diff_neg=-1.0*time_diff;
  time_diff_neg_degital = time_diff /RESOLUTION;
end

endmodule

