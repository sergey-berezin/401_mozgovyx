dotnet sln remove YoloPackage
dotnet remove ConsoleApp reference ./YoloPackage/YoloPackage.csproj
dotnet add ConsoleApp package Mozgovyx.YoloPackage --version 2.1.2709.1
