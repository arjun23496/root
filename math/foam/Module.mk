# Module.mk for foam module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := foam
MODDIR       := $(ROOT_SRCDIR)/math/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

FOAMDIR      := $(MODDIR)
FOAMDIRS     := $(FOAMDIR)/src
FOAMDIRI     := $(FOAMDIR)/inc

##### libFoam.so #####
FOAML      := $(MODDIRI)/LinkDef.h
FOAMDS     := $(call stripsrc,$(MODDIRS)/G__Foam.cxx)
FOAMDO     := $(FOAMDS:.cxx=.o)
FOAMDH     := $(FOAMDS:.cxx=.h)

FOAMH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
FOAMS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
FOAMO      := $(call stripsrc,$(FOAMS:.cxx=.o))

FOAMDEP    := $(FOAMO:.o=.d) $(FOAMDO:.o=.d)

FOAMLIB    := $(LPATH)/libFoam.$(SOEXT)
FOAMMAP    := $(FOAMLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
FOAMH_REL   := $(patsubst $(MODDIRI)/%.h,include/%.h,$(FOAMH))
ALLHDRS     += $(FOAMH_REL)
ALLLIBS     += $(FOAMLIB)
ALLMAPS     += $(FOAMMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(FOAMH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Math_Foam { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(FOAMLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(FOAMDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(FOAMDIRI)/%.h
		cp $< $@

$(FOAMLIB):     $(FOAMO) $(FOAMDO) $(ORDER_) $(MAINLIBS) $(FOAMLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libFoam.$(SOEXT) $@ "$(FOAMO) $(FOAMDO)" \
		   "$(FOAMLIBEXTRA)"

$(call pcmrule,FOAM)
	$(noop)

$(FOAMDS):      $(FOAMH) $(FOAML) $(ROOTCLINGEXE) $(call pcmdep,FOAM)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,FOAM) -c $(FOAMH) $(FOAML)

$(FOAMMAP):     $(FOAMH) $(FOAML) $(ROOTCLINGEXE) $(call pcmdep,FOAM)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(FOAMDS) $(call dictModule,FOAM) -c $(FOAMH) $(FOAML)

all-$(MODNAME): $(FOAMLIB)

clean-$(MODNAME):
		@rm -f $(FOAMO) $(FOAMDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(FOAMDEP) $(FOAMDS) $(FOAMDH) $(FOAMLIB) $(FOAMMAP)

distclean::     distclean-$(MODNAME)
