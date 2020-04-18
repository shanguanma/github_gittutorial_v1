#!/usr/bin/env perl
#use warnings; #sed replacement for -w perl parameter
#use strict;
# subrotine definition
sub is_value_exist_in_array{
    # first argument:string value to be searched in array
    $value = $_[0];
    # second argument : refrence to array to be searched in
    $arrayref = $_[1];
    # creat the array by dereferencing
    @my_array = @$arrayref;
    
    $result = "does not exist in";
    foreach $element (@my_array){
        if ($element eq $value) {
            $result = "exist in";
            last;
         }
      
    }
    # print result
    print "value $value $result array [ @my_array ]\n";
   

}

# subroutine call
@foo = ('we', 'are',5,'happy','perl','programmers');
$foo_arrayref = \@foo;

print "we are calling the subroutine is_value_exists_in_array() now\n";
is_value_exists_in_array('hello',$foo_arrayref);
print "we are calling the subroutine is_value_exists_in_array() again \n";
is_value_exists_in_array('happy',$foo_arrayref);
