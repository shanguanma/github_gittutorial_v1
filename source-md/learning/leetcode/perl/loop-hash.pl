#!/usr/bin/perl

%colors = ("a" => 1, "b"=>2, "c"=>3, "d"=>4, "e"=>5);
foreach (sort keys %colors)
{
    print $colors{$_} . "\n";
}
